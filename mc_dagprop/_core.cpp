// mc_dagprop/_core.cpp

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <_custom_rng.hpp>

namespace py = pybind11;
using namespace std;

// ── Aliases ───────────────────────────────────────────────────────────────
using NodeIndex = int;
using EdgeIndex = int;
using Preds = vector<pair<NodeIndex, EdgeIndex>>;
using RNG = utl::random::generators::Xoshiro256PP;

// ── Hash for pair<int,int> ────────────────────────────────────────────────
namespace std {
template <typename A, typename B>
struct hash<pair<A, B>> {
    size_t operator()(pair<A, B> const &p) const noexcept { return hash<A>{}(p.first) ^ (hash<B>{}(p.second) << 1); }
};
}  // namespace std

// ── Core Data Types ───────────────────────────────────────────────────────
struct EventTimestamp {
    double earliest, latest, actual;
};

struct SimEvent {
    std::string node_id;
    EventTimestamp ts;
};

struct SimActivity {
    double duration;
    int activity_type;
};

// ── Simulation Context ───────────────────────────────────────────────────
struct SimContext {
    vector<SimEvent> events;
    // user provides link_index + SimActivity
    unordered_map<pair<NodeIndex, NodeIndex>, pair<EdgeIndex, SimActivity>> activity_map;
    vector<pair<NodeIndex, Preds>> precedence_list;
    double max_delay;

    SimContext(vector<SimEvent> ev, unordered_map<pair<NodeIndex, NodeIndex>, pair<EdgeIndex, SimActivity>> am,
               vector<pair<NodeIndex, Preds>> pl, double md)
        : events(move(ev)), activity_map(move(am)), precedence_list(move(pl)), max_delay(md) {}
};

// ── Simulation Result ────────────────────────────────────────────────────
struct SimResult {
    vector<double> realized;
    vector<double> durations;
    vector<NodeIndex> cause_event;
};

// ── Delay Distributions ──────────────────────────────────────────────────
struct ConstantDist {
    double factor;
    ConstantDist(const double f = 0.0) : factor(f) {}
    double sample(RNG &, const double d) const { return d * factor; }
};

struct ExponentialDist {
    double lambda, max_scale;
    exponential_distribution<double> dist;
    ExponentialDist(const double lam = 1.0, const double mx = 1.0) : lambda(lam), max_scale(mx), dist(1.0 / lam) {}
    double sample(RNG &rng, const double d) const {
        double x;
        do {
            x = dist(rng);
        } while (x > max_scale);
        return x * d;
    }
};

struct GammaDist {
    double shape, scale, max_scale;
    gamma_distribution<double> dist;
    GammaDist(const double k = 1.0, const double s = 1.0, const double m = numeric_limits<double>::infinity())
        : shape(k), scale(s), max_scale(m), dist(k, s) {}
    double sample(RNG &rng, const double d) const {
        double x;
        do {
            x = dist(rng);
        } while (x > max_scale);
        return x * d;
    }
};

// ── Empirical “table” distributions ─────────────────────────────────────

// 1) Absolute: user‐supplied values are taken literally
struct EmpiricalAbsoluteDist {
    std::vector<double> values;
    std::discrete_distribution<size_t> dist;

    EmpiricalAbsoluteDist(
      std::vector<double> vals,
      std::vector<double> weights
    )
      : values(std::move(vals))
      // build the distribution *after* we’ve checked sizes
      , dist()
    {
        if (values.size() != weights.size())
            throw std::runtime_error("EmpiricalAbsoluteDist: values and weights must have same length");
        dist = std::discrete_distribution<size_t>(weights.begin(), weights.end());
    }

    double sample(RNG &rng, double /*duration*/) const {
        return values[ dist(rng) ];
    }
};

// 2) Relative: user‐supplied factors in [0..∞), multiplied by the base duration
struct EmpiricalRelativeDist {
    std::vector<double> factors;
    std::discrete_distribution<size_t> dist;

    EmpiricalRelativeDist(std::vector<double> facs, std::vector<double> weights): factors(std::move(facs)), dist()
    {
        if (factors.size() != weights.size())
            throw std::runtime_error("EmpiricalRelativeDist: factors and weights must have same length");
        dist = std::discrete_distribution<size_t>(weights.begin(), weights.end());
    }

    double sample(RNG &rng, double duration) const {
        return factors[ dist(rng) ] * duration;
    }
};

using DistVar = std::variant<
    ConstantDist,
    ExponentialDist,
    GammaDist,
    EmpiricalAbsoluteDist,
    EmpiricalRelativeDist
>;

// ── Delay Generator ──────────────────────────────────────────────────────
class GenericDelayGenerator {
   public:
    RNG rng_;
    unordered_map<int, DistVar> dist_map_;

    GenericDelayGenerator() : rng_(random_device{}()) {}

    void set_seed(int s) { rng_.seed(s); }
    void add_constant(int t, double f) { dist_map_[t] = ConstantDist{f}; }
    void add_exponential(int t, double lam, double mx) { dist_map_[t] = ExponentialDist{lam, mx}; }
    void add_gamma(int t, double k, double s, double m = numeric_limits<double>::infinity()) {
        dist_map_[t] = GammaDist{k, s, m};
    }
};

// ── Simulator ────────────────────────────────────────────────────────────
class Simulator {
    SimContext ctx_;
    vector<DistVar> dists_;           // flattened distributions
    vector<SimActivity> activities_;  // length = max_link_index+1
    vector<int> act2dist_;            // same length, -1=no-dist
    RNG rng_;
    vector<Preds> preds_by_node_;
    vector<NodeIndex> node_indices_;  // for each event, its index

    // scratch buffers
    vector<double> earliest_allowed_, extended_durations_, realized_ts_;
    vector<NodeIndex> cause_;

   public:
    Simulator(SimContext c, GenericDelayGenerator gen) : ctx_(move(c)), rng_(random_device{}()) {
        // 1) flatten distributions -> dists_, build type→dist-index
        unordered_map<int, int> type2idx;
        dists_.reserve(gen.dist_map_.size());
        auto di = 0;
        for (auto &kv : gen.dist_map_) {
            type2idx[kv.first] = di;
            dists_.push_back(kv.second);
            ++di;
        }

        // 2) figure out how many links & allocate activities_ & act2dist_
        auto max_link = 0;
        for (auto &kv : ctx_.activity_map) {
            max_link = max(max_link, kv.second.first);
        }
        if (max_link != int(ctx_.activity_map.size()) - 1) {
            throw runtime_error(
                "Mismatch between link index and activity map size check your "
                "activity_map, indices should be 0 to N-1");
        }
        const auto L = max_link + 1;
        activities_.assign(L, SimActivity{0.0, -1});
        act2dist_.assign(L, -1);

        // fill in by link_index
        for (auto &kv : ctx_.activity_map) {
            auto link_idx = kv.second.first;
            const auto &activity = kv.second.second;
            activities_[link_idx] = activity;
            auto it = type2idx.find(activity.activity_type);
            if (it != type2idx.end()) {
                act2dist_[link_idx] = it->second;
            }
        }

        // 3) build preds_by_node_
        auto N = int(ctx_.events.size());
        preds_by_node_.assign(N, {});
        node_indices_.resize(N);
        auto i = 0;
        for (auto &p : ctx_.precedence_list) {
            for (auto &pr : p.second) {
                preds_by_node_[p.first].push_back(pr);
            }
            node_indices_[i] = p.first;
            ++i;
        }

        // 4) scratch buffers
        earliest_allowed_.resize(N);
        realized_ts_.resize(N);
        cause_.resize(N);
        extended_durations_.resize(L);
    }

    inline const int node_count() const noexcept { return int(earliest_allowed_.size()); }
    inline const int activity_count() const noexcept { return int(extended_durations_.size()); }

    const SimResult run(const int seed) {
        rng_.seed(seed);
        const int N = node_count(), activity_count_ = activity_count();

        // load earliest & scheduled
        for (auto i = 0; i < N; ++i) {
            realized_ts_[i] = ctx_.events[i].ts.earliest;
        }
        // base durations
        for (auto i = 0; i < activity_count_; ++i) {
            extended_durations_[i] = activities_[i].duration;
        }
        // sample + compound
        for (auto i = 0; i < activity_count_; ++i) {
            auto di = act2dist_[i];
            if (di >= 0) {
                auto delta = extended_durations_[i];
                // sample from distribution
                auto extra = visit([&](auto &d) { return d.sample(rng_, delta); }, dists_[di]);
                if (extra > 0.0) {
                    extended_durations_[i] = delta + extra;
                }
            }
        }
        // init propagate
        std::fill(cause_.begin(), cause_.end(), -1);

        // actual propagation
        for (auto i = 0; i < N; ++i) {
            auto n_index = node_indices_[i];
            auto latest = realized_ts_[n_index];
            NodeIndex cause = n_index;
            auto changed = false;
            for (auto &pr : preds_by_node_[n_index]) {
                auto t = realized_ts_[pr.first] + extended_durations_[pr.second];
                if (t >= latest) {
                    latest = t;
                    cause = pr.first;
                    changed = true;
                }
            }
            if (changed) {
                realized_ts_[n_index] = latest;
                cause_[n_index] = cause;
            }
        }

        return SimResult{realized_ts_, extended_durations_, cause_};
    }

    vector<SimResult> run_many(const vector<int> &seeds) {
        vector<SimResult> out;
        out.reserve(seeds.size());
        for (auto s : seeds) {
            out.push_back(run(s));
        }
        return out;
    }
};

// ── Python Bindings ─────────────────────────────────────────────────────
PYBIND11_MODULE(_core, m) {
    m.doc() = "Core Monte-Carlo DAG-propagation simulator";

    // EventTimestamp
    py::class_<EventTimestamp>(m, "EventTimestamp")
        .def(py::init<double, double, double>(), py::arg("earliest"), py::arg("latest"), py::arg("actual"),
             "Create an event timestamp (earliest, latest, actual).")
        .def_readwrite("earliest", &EventTimestamp::earliest, "Earliest bound")
        .def_readwrite("latest", &EventTimestamp::latest, "Latest bound")
        .def_readwrite("actual", &EventTimestamp::actual, "Scheduled time");

    // SimEvent
    py::class_<SimEvent>(m, "SimEvent")
        .def(py::init<std::string, EventTimestamp>(), py::arg("node_id"), py::arg("timestamp"),
             "An event node with its ID and timestamp")
        .def_readwrite("node_id", &SimEvent::node_id, "Node identifier")
        .def_readwrite("timestamp", &SimEvent::ts, "Event timing info");

    // SimActivity
    py::class_<SimActivity>(m, "SimActivity")
        .def(py::init<double, int>(), py::arg("minimal_duration"), py::arg("activity_type"),
             "An activity (edge) with base duration and type")
        .def_readwrite("minimal_duration", &SimActivity::duration, "Base duration")
        .def_readwrite("activity_type", &SimActivity::activity_type, "Type ID for delay dist.");

    // SimContext
    py::class_<SimContext>(m, "SimContext")
        .def(py::init<vector<SimEvent>, unordered_map<pair<NodeIndex, NodeIndex>, pair<EdgeIndex, SimActivity>>,
                      vector<pair<NodeIndex, Preds>>, double>(),
             py::arg("events"), py::arg("activities"), py::arg("precedence_list"), py::arg("max_delay"),
             "Wraps a DAG: events, activity_map, precedence_list, max_delay")
        .def_readwrite("events", &SimContext::events)
        .def_readwrite("activities", &SimContext::activity_map)
        .def_readwrite("precedence_list", &SimContext::precedence_list)
        .def_readwrite("max_delay", &SimContext::max_delay);

    // SimResult // SimResult → return NumPy arrays instead of lists
    py::class_<SimResult>(m, "SimResult")
        .def_property_readonly(
            "realized",
            [](const SimResult &r) {
                // make a new 1-D array and copy C++ vector → NumPy
                py::array_t<double> arr(r.realized.size());
                double *ptr = arr.mutable_data();
                for (size_t i = 0; i < r.realized.size(); ++i) ptr[i] = r.realized[i];
                return arr;
            },
            "Final event times as a NumPy array")
        .def_property_readonly(
            "durations",
            [](const SimResult &r) {
                py::array_t<double> arr(r.durations.size());
                double *ptr = arr.mutable_data();
                for (size_t i = 0; i < r.durations.size(); ++i) ptr[i] = r.durations[i];
                return arr;
            },
            "Per-link durations (incl. extra) as a NumPy array")
        .def_property_readonly(
            "cause_event",
            [](const SimResult &r) {
                py::array_t<int> arr(r.cause_event.size());
                int *ptr = arr.mutable_data();
                for (size_t i = 0; i < r.cause_event.size(); ++i) ptr[i] = r.cause_event[i];
                return arr;
            },
            "Index of predecessor causing each event as a NumPy array");

    // GenericDelayGenerator
    py::class_<GenericDelayGenerator>(m, "GenericDelayGenerator")
        .def(py::init<>(), "Create a new delay‐generator")
        .def("set_seed", &GenericDelayGenerator::set_seed, py::arg("seed"), "Set RNG seed for reproducibility")
        .def("add_constant", &GenericDelayGenerator::add_constant, py::arg("activity_type"), py::arg("factor"),
             "Constant: delay = factor * duration")
        .def("add_exponential", &GenericDelayGenerator::add_exponential, py::arg("activity_type"), py::arg("lambda_"),
             py::arg("max_scale"), "Exponential(λ) truncated at max_scale")
        .def("add_gamma", &GenericDelayGenerator::add_gamma, py::arg("activity_type"), py::arg("shape"),
             py::arg("scale"), py::arg("max_scale") = numeric_limits<double>::infinity(),
             "Gamma(shape,scale) truncated at max_scale")
        .def("add_empirical_absolute", [](GenericDelayGenerator &g,
            int activity_type,
            std::vector<double> values,
            std::vector<double> weights)
         { g.dist_map_[activity_type] = EmpiricalAbsoluteDist{std::move(values), std::move(weights)}; },
         py::arg("activity_type"),
         py::arg("values"),
         py::arg("weights"),
         "Empirical absolute: draw one of your provided values, weighted by weights.")
        .def("add_empirical_relative",
         [](GenericDelayGenerator &g,
            int activity_type,
            std::vector<double> factors,
            std::vector<double> weights)
         { g.dist_map_[activity_type] = EmpiricalRelativeDist{std::move(factors), std::move(weights)}; },
         py::arg("activity_type"),
         py::arg("factors"),
         py::arg("weights"),
         "Empirical relative: draw a factor in [0,∞), then multiply by the activity duration.");

    // Simulator
    py::class_<Simulator>(m, "Simulator")
        .def(py::init<SimContext, GenericDelayGenerator>(), py::arg("context"), py::arg("generator"),
             "Construct simulator with context and delay‐generator")
        .def("node_count", &Simulator::node_count, "Number of events")
        .def("activity_count", &Simulator::activity_count, "Number of links")
        .def("run", &Simulator::run, py::arg("seed"), "Run single sim")
        .def("run_many", &Simulator::run_many, py::arg("seeds"), "Run batch sims");
}
