// mc_dagprop/_core.cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using namespace std;

// ── Aliases ───────────────────────────────────────────────────────────────
using NodeIndex = int;
using EdgeIndex = int;
using Preds = vector<pair<NodeIndex, EdgeIndex>>;

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
    unordered_map<pair<NodeIndex, NodeIndex>, SimActivity> activities;
    vector<pair<NodeIndex, Preds>> precedence_list;
    double max_delay;

    SimContext(vector<SimEvent> ev, unordered_map<pair<NodeIndex, NodeIndex>, SimActivity> am,
               vector<pair<NodeIndex, Preds>> pl, double md)
        : events(move(ev)), activities(move(am)), precedence_list(move(pl)), max_delay(md) {}
};

// ── Simulation Result ────────────────────────────────────────────────────
struct SimResult {
    vector<double> realized;
    vector<double> delays;
    vector<int> cause_event;
};

// ── Delay Distributions ──────────────────────────────────────────────────
struct ConstantDist {
    double factor;
    ConstantDist(double f = 0.0) : factor(f) {}
    double sample(mt19937 &, double d) const { return d * factor; }
};

struct ExponentialDist {
    double lambda, max_scale;
    exponential_distribution<double> dist;
    ExponentialDist(double lam = 1.0, double mx = 1.0) : lambda(lam), max_scale(mx), dist(1.0 / lam) {}
    double sample(mt19937 &rng, double d) const {
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
    GammaDist(double k = 1.0, double s = 1.0, double m = numeric_limits<double>::infinity())
        : shape(k), scale(s), max_scale(m), dist(k, s) {}
    double sample(mt19937 &rng, double d) const {
        double x;
        do {
            x = dist(rng);
        } while (x > max_scale);
        return x * d;
    }
};

using DistVar = variant<ConstantDist, ExponentialDist, GammaDist>;

// ── Delay Generator ──────────────────────────────────────────────────────
class GenericDelayGenerator {
   public:
    mt19937 rng_;
    unordered_map<int, DistVar> dist_map_;

    GenericDelayGenerator() : rng_(random_device{}()) {}

    void set_seed(int s) { rng_.seed(s); }

    void add_constant(int t, double f) { dist_map_[t] = ConstantDist{f}; }
    void add_exponential(int t, double lam, double mx) { dist_map_[t] = ExponentialDist{lam, mx}; }
    void add_gamma(int t, double k, double s, double m = numeric_limits<double>::infinity()) {
        dist_map_[t] = GammaDist{k, s, m};
    }
};

// ── Simulator ───────────────────────────────────────────────────────────
class Simulator {
    SimContext ctx_;
    vector<DistVar> dists_;
    vector<SimActivity> acts_;
    vector<int> act2dist_;
    mt19937 rng_;
    vector<Preds> preds_by_node_;

    // scratch buffers (pre-allocated)
    vector<double> earliest_allowed_, sched_, extended_durations_, realized_event_ts_;
    vector<int> cause_;

   public:
    Simulator(SimContext c, GenericDelayGenerator gen) : ctx_(move(c)), rng_(random_device{}()) {
        // 1) flatten distributions
        unordered_map<int, int> type2idx;
        dists_.reserve(gen.dist_map_.size());
        int dist_index = 0;
        for (auto &kv : gen.dist_map_) {
            type2idx[kv.first] = dist_index;
            dists_.push_back(kv.second);
            ++dist_index;
        }
        // 2) flatten activities and map to dist index
        acts_.reserve(ctx_.activities.size());
        act2dist_.reserve(ctx_.activities.size());
        for (auto &kv : ctx_.activities) {
            acts_.push_back(kv.second);
            auto it = type2idx.find(kv.second.activity_type);
            act2dist_.push_back(it == type2idx.end() ? -1 : it->second);
        }
        // 3) build per-node preds
        int N = (int)ctx_.events.size();
        preds_by_node_.assign(N, {});
        for (auto &p : ctx_.precedence_list) {
            auto &v = preds_by_node_[p.first];
            for (auto &pr : p.second) v.push_back(pr);
        }
        // 4) scratch
        earliest_allowed_.resize(N);
        sched_.resize(N);
        realized_event_ts_.resize(N);
        cause_.resize(N);
        extended_durations_.resize(acts_.size());
    }

    // convenience accessors
    int node_count() const noexcept { return (int)earliest_allowed_.size(); }
    int activity_count() const noexcept { return (int)extended_durations_.size(); }

    // single run
    SimResult run(int seed) {
        rng_.seed(seed);
        const auto N = node_count();
        const auto M = activity_count();
        // load timestamps
        // TODO: Maybe there are more efficient iterators here?
        for (auto i = 0; i < N; ++i) {
            earliest_allowed_[i] = ctx_.events[i].ts.earliest;
            sched_[i] = ctx_.events[i].ts.actual;
        }
        // TODO: Maybe there are more efficient iterators here?
        for (auto i = 0; i < M; ++i) {
            extended_durations_[i] = acts_[i].duration;
        } 
        // sample + compound
        for (auto i = 0; i < M; ++i) {
            double extra = 0.0;
            auto dist_index = act2dist_[i];
            auto delta = extended_durations_[i];
            // TODO: here we might have something more efficient? This is functional code, might not be fast?
            if (dist_index >= 0) {
                auto &d = dists_[dist_index];
                extra = visit([&](auto &x) { return x.sample(rng_, delta); }, d);
            }
            if (extra > 0.0) {
                extended_durations_[i] = delta + extra;
            }
        }
        // init
        copy(earliest_allowed_.begin(), earliest_allowed_.end(), realized_event_ts_.begin());
        iota(cause_.begin(), cause_.end(), -1);
        // propagate
        for (auto node = 0; node < N; ++node) {
            auto t_0 = realized_event_ts_[node];
            auto this_cause = node;
            for (auto &pr : preds_by_node_[node]) {
                auto t = realized_event_ts_[pr.first] + extended_durations_[pr.second];
                if (t > t_0) {
                    t_0 = t;
                    this_cause = pr.first;
                }
            }
            auto is_delayed = this_cause != node;
            if (is_delayed)  {
                realized_event_ts_[node] = t_0;
                cause_[node] = this_cause;
            }
        }
        return SimResult{realized_event_ts_, extended_durations_, cause_};
    }

    // old: per‐run vector<SimResult>
    vector<SimResult> run_many(const vector<int> &seeds) {
        vector<SimResult> out;
        out.reserve(seeds.size());
        for (auto s : seeds) out.push_back(run(s));
        return out;
    }
};

// ── Python Bindings ─────────────────────────────────────────────────────
PYBIND11_MODULE(_core, m) {
    m.doc() = "Core Monte-Carlo DAG-propagation simulator";

    // EventTimestamp
    py::class_<EventTimestamp>(m, "EventTimestamp")
        .def(py::init<double, double, double>(), py::arg("earliest"), py::arg("latest"), py::arg("actual"))
        .def_readwrite("earliest", &EventTimestamp::earliest)
        .def_readwrite("latest", &EventTimestamp::latest)
        .def_readwrite("actual", &EventTimestamp::actual);

    // SimEvent
    py::class_<SimEvent>(m, "SimEvent")
        .def(py::init<std::string, EventTimestamp>(), py::arg("node_id"), py::arg("timestamp"))
        .def_readwrite("node_id", &SimEvent::node_id)
        .def_readwrite("timestamp", &SimEvent::ts);

    // SimActivity
    py::class_<SimActivity>(m, "SimActivity")
        .def(py::init<double, int>(), py::arg("minimal_duration"), py::arg("activity_type"))
        .def_readwrite("minimal_duration", &SimActivity::duration)
        .def_readwrite("activity_type", &SimActivity::activity_type);

    // SimContext
    py::class_<SimContext>(m, "SimContext")
        .def(py::init<vector<SimEvent>, unordered_map<pair<int, int>, SimActivity>, vector<pair<int, Preds>>, double>(),
             py::arg("events"), py::arg("activities"), py::arg("precedence_list"), py::arg("max_delay"))
        .def_readwrite("events", &SimContext::events)
        .def_readwrite("activities", &SimContext::activities)
        .def_readwrite("precedence_list", &SimContext::precedence_list)
        .def_readwrite("max_delay", &SimContext::max_delay);

    // SimResult
    py::class_<SimResult>(m, "SimResult")
        .def_readonly("realized", &SimResult::realized)
        .def_readonly("delays", &SimResult::delays)
        .def_readonly("cause_event", &SimResult::cause_event);

    // DelayGenerator
    py::class_<GenericDelayGenerator>(m, "GenericDelayGenerator")
        .def(py::init<>())
        .def("set_seed", &GenericDelayGenerator::set_seed, py::arg("seed"))
        .def("add_constant", &GenericDelayGenerator::add_constant, py::arg("activity_type"), py::arg("factor"))
        .def("add_exponential", &GenericDelayGenerator::add_exponential, py::arg("activity_type"), py::arg("lambda_"),
             py::arg("max_scale"))
        .def("add_gamma", &GenericDelayGenerator::add_gamma, py::arg("activity_type"), py::arg("shape"),
             py::arg("scale"), py::arg("max_scale") = numeric_limits<double>::infinity());

    // Simulator
    py::class_<Simulator>(m, "Simulator")
        .def(py::init<SimContext, GenericDelayGenerator>(), py::arg("context"), py::arg("generator"))
        .def("node_count", &Simulator::node_count)
        .def("activity_count", &Simulator::activity_count)
        .def("run", &Simulator::run, py::arg("seed"))
        .def("run_many", &Simulator::run_many, py::arg("seeds"))
        .def(
            "run_many_arrays",
            [](Simulator &sim, const std::vector<int> &seeds) {
                int R = (int)seeds.size();
                int N = sim.node_count();
                int M = sim.activity_count();

                // allocate 2D arrays with shape (N, R) and (M, R)
                auto A = py::array_t<double>({N, R});
                auto B = py::array_t<double>({M, R});
                auto C = py::array_t<int>({N, R});

                double *Ap = A.mutable_data();
                double *Bp = B.mutable_data();
                int *Cp = C.mutable_data();

                // for each run, fill column i
                for (int i = 0; i < R; ++i) {
                    auto res = sim.run(seeds[i]);
                    // realized times → A[:, i]
                    for (int n = 0; n < N; ++n) Ap[n * R + i] = res.realized[n];
                    // delays → B[:, i]
                    for (int e = 0; e < M; ++e) Bp[e * R + i] = res.delays[e];
                    // cause_event → C[:, i]
                    for (int n = 0; n < N; ++n) Cp[n * R + i] = res.cause_event[n];
                }

                return py::make_tuple(A, B, C);
            },
            py::arg("seeds"),
            "Run batch and return three 2D arrays:\n"
            "- realized[N, R]\n"
            "- delays[M, R]\n"
            "- cause_event[N, R]\n");
}
