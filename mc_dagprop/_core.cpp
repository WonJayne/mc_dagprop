// mc_dagprop/_core.cpp

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <_custom_rng.hpp>
#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

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

    EmpiricalAbsoluteDist(std::vector<double> vals, std::vector<double> weights)
        : values(std::move(vals))
          // build the distribution *after* we’ve checked sizes
          ,
          dist() {
        if (values.size() != weights.size())
            throw std::runtime_error("EmpiricalAbsoluteDist: values and weights must have same length");
        dist = std::discrete_distribution<size_t>(weights.begin(), weights.end());
    }

    double sample(RNG &rng, double /*duration*/) const { return values[dist(rng)]; }
};

// 2) Relative: user‐supplied factors in [0..∞), multiplied by the base duration
struct EmpiricalRelativeDist {
    std::vector<double> factors;
    std::discrete_distribution<size_t> dist;

    EmpiricalRelativeDist(std::vector<double> facs, std::vector<double> weights) : factors(std::move(facs)), dist() {
        if (factors.size() != weights.size())
            throw std::runtime_error("EmpiricalRelativeDist: factors and weights must have same length");
        dist = std::discrete_distribution<size_t>(weights.begin(), weights.end());
    }

    double sample(RNG &rng, double duration) const { return factors[dist(rng)] * duration; }
};

using DistVar = std::variant<ConstantDist, ExponentialDist, GammaDist, EmpiricalAbsoluteDist, EmpiricalRelativeDist>;

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
    SimContext context_;

    // Delay distributions (flattened)
    std::vector<DistVar> delay_distributions_;
    std::unordered_map<int, int> activity_type_to_dist_index_;

    // Activities: one per link index
    std::vector<SimActivity> activities_;
    std::vector<int> activity_to_dist_index_;  // -1 = no delay

    // Precedence (CSR format)
    std::vector<NodeIndex> flat_predecessor_sources_;  // all predecessor node indices
    std::vector<EdgeIndex> flat_predecessor_edges_;    // corresponding edge indices
    std::vector<size_t> predecessor_offsets_;         // prefix offsets per event
    std::vector<NodeIndex> event_evaluation_order_;    // order in which to process events

    // RNG
    RNG rng_;

    // Sampler function: signature(sample_rng, distribution_variant, base_duration)
    using SamplerFunc = double(*)(RNG&, const DistVar&, double);
    std::vector<SamplerFunc> sampler_functions_;

    // Scratch buffers
    std::vector<double> earliest_times_;
    std::vector<double> actual_durations_;
    std::vector<double> realized_times_;
    std::vector<NodeIndex> causing_event_index_;

public:
    Simulator(SimContext context, GenericDelayGenerator generator)
        : context_(std::move(context)), rng_(std::random_device{}()) {
        // 0) Validate reserved activity_type
        if (generator.dist_map_.count(-1)) {
            throw std::runtime_error("Activity type -1 is reserved for no delay");
        }

        // 1) Flatten distributions and build type->index map
        delay_distributions_.reserve(generator.dist_map_.size());
        int dist_counter = 0;
        for (auto &entry : generator.dist_map_) {
            activity_type_to_dist_index_[entry.first] = dist_counter;
            delay_distributions_.push_back(entry.second);
            ++dist_counter;
        }

        // 2) Allocate activities and map each link to a distribution index
        int max_link_index = -1;
        for (auto &kv : context_.activity_map) {
            max_link_index = std::max(max_link_index, kv.second.first);
        }
        int link_count = max_link_index + 1;
        activities_.assign(link_count, SimActivity{0.0, -1});
        activity_to_dist_index_.assign(link_count, -1);

        for (auto &kv : context_.activity_map) {
            EdgeIndex link_idx = kv.second.first;
            activities_[link_idx] = kv.second.second;
            auto it = activity_type_to_dist_index_.find(activities_[link_idx].activity_type);
            if (it != activity_type_to_dist_index_.end()) {
                activity_to_dist_index_[link_idx] = it->second;
            }
        }

        // 3) Build CSR for precedences
        int event_count = int(context_.events.size());
        predecessor_offsets_.assign(event_count + 1, 0);
        for (auto &pr : context_.precedence_list) {
            NodeIndex event_id = pr.first;
            predecessor_offsets_[event_id + 1] = pr.second.size();
        }
        for (int i = 1; i <= event_count; ++i) {
            predecessor_offsets_[i] += predecessor_offsets_[i - 1];
        }
        int total_predecessors = predecessor_offsets_[event_count];
        flat_predecessor_sources_.resize(total_predecessors);
        flat_predecessor_edges_.resize(total_predecessors);

        std::vector<size_t> write_positions = predecessor_offsets_;
        event_evaluation_order_.reserve(event_count);
        for (auto &entry : context_.precedence_list) {
            NodeIndex event_id = entry.first;
            event_evaluation_order_.push_back(event_id);
            for (auto &pr : entry.second) {
                size_t idx = write_positions[event_id]++;
                flat_predecessor_sources_[idx] = pr.first;
                flat_predecessor_edges_[idx] = pr.second;
            }
        }

        // 4) Prepare sampler function pointers
        sampler_functions_.resize(link_count, nullptr);
        for (int link = 0; link < link_count; ++link) {
            int dist_idx = activity_to_dist_index_[link];
            if (dist_idx < 0) continue;
            sampler_functions_[link] = [](RNG &rng, const DistVar &var, double base_dur) {
                return std::visit([&](auto &dist) { return dist.sample(rng, base_dur); }, var);
            };
        }

        // 5) Allocate scratch buffers
        earliest_times_.resize(event_count);
        realized_times_.resize(event_count);
        causing_event_index_.assign(event_count, -1);
        actual_durations_.resize(link_count);

        for (int link = 0; link < link_count; ++link) {
            if (activity_to_dist_index_[link] < 0)
                actual_durations_[link] = activities_[link].duration;
        }
    }

    inline int node_count() const noexcept { return int(earliest_times_.size()); }
    inline int activity_count() const noexcept { return int(actual_durations_.size()); }

    SimResult run(int seed) {
        rng_.seed(seed);
        int E = node_count();
        int L = activity_count();

        // Load earliest times
        for (int i = 0; i < E; ++i) {
            realized_times_[i] = context_.events[i].ts.earliest;
        }

        // Sample delays
        for (int link = 0; link < L; ++link) {
            int dist_idx = activity_to_dist_index_[link];
            if (dist_idx < 0) continue;
            double base_dur = activities_[link].duration;
            double extra = sampler_functions_[link](rng_, delay_distributions_[dist_idx], base_dur);
            actual_durations_[link] = base_dur + extra;
        }

        // Propagate events
        for (NodeIndex event_id : event_evaluation_order_) {
            double latest = realized_times_[event_id];
            NodeIndex cause = -1;
            for (size_t idx = predecessor_offsets_[event_id]; idx < predecessor_offsets_[event_id + 1]; ++idx) {
                NodeIndex src = flat_predecessor_sources_[idx];
                EdgeIndex edge = flat_predecessor_edges_[idx];
                double t = realized_times_[src] + actual_durations_[edge];
                if (t >= latest) {
                    latest = t;
                    cause = src;
                }
            }
            realized_times_[event_id] = latest;
            causing_event_index_[event_id] = cause;
        }

        return SimResult{realized_times_, actual_durations_, causing_event_index_};
    }

    std::vector<SimResult> run_many(const std::vector<int> &seeds) {
        py::gil_scoped_release release;
        std::vector<SimResult> results;
        results.reserve(seeds.size());
        for (int s : seeds) results.emplace_back(run(s));
        return results;
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
    py::class_<SimResult>(m, "SimResult", py::buffer_protocol())
        .def_buffer([](SimResult &r) -> py::buffer_info {
            return py::buffer_info(r.realized.data(),                        // Pointer to buffer
                                   sizeof(double),                           // Size of one scalar
                                   py::format_descriptor<double>::format(),  // Python struct-style format descriptor
                                   1,                                        // Number of dimensions
                                   {r.realized.size()},                      // Buffer dimensions
                                   {sizeof(double)}                          // Strides (in bytes) for each index
            );
        })
        .def_property_readonly(
            "realized",
            [](const SimResult &r) {
                return py::array(r.realized.size(),  // shape
                                 r.realized.data(),  // pointer to data
                                 py::cast(r)         // capsule to ensure SimResult stays alive
                );
            },
            "Final event times as a NumPy array")
        .def_property_readonly(
            "durations", [](SimResult &r) { return py::array(r.durations.size(), r.durations.data(), py::cast(r)); },
            "Per-link durations (incl. extra) as a NumPy array")
        .def_property_readonly(
            "cause_event",
            [](SimResult &r) { return py::array(r.cause_event.size(), r.cause_event.data(), py::cast(r)); },
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
        .def(
            "add_empirical_absolute",
            [](GenericDelayGenerator &g, int activity_type, std::vector<double> values, std::vector<double> weights) {
                g.dist_map_[activity_type] = EmpiricalAbsoluteDist{std::move(values), std::move(weights)};
            },
            py::arg("activity_type"), py::arg("values"), py::arg("weights"),
            "Empirical absolute: draw one of your provided values, weighted by weights.")
        .def(
            "add_empirical_relative",
            [](GenericDelayGenerator &g, int activity_type, std::vector<double> factors, std::vector<double> weights) {
                g.dist_map_[activity_type] = EmpiricalRelativeDist{std::move(factors), std::move(weights)};
            },
            py::arg("activity_type"), py::arg("factors"), py::arg("weights"),
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
