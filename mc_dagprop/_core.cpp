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

using namespace std;
namespace py = pybind11;

// Bring your alias into scope for pybind:
using NodeIndex = int;
using EdgeIndex = int;
using Preds = vector<pair<NodeIndex, EdgeIndex>>;

// ─── custom hash for pair ─────────────────────────────────────────────
namespace std {
template <typename A, typename B>
struct hash<pair<A, B>> {
    size_t operator()(pair<A, B> const& p) const noexcept { return hash<A>{}(p.first) ^ (hash<B>{}(p.second) << 1); }
};
}  // namespace std

// ─── data types ────────────────────────────────────────────────────────
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

// ─── simulation context ───────────────────────────────────────────────
struct SimContext {
    vector<SimEvent> events;
    unordered_map<pair<NodeIndex, NodeIndex>, SimActivity> activities;
    vector<pair<NodeIndex, Preds>> precedence_list;
    double max_delay;

    SimContext(vector<SimEvent> ev, unordered_map<pair<NodeIndex, NodeIndex>, SimActivity> am,
               vector<pair<NodeIndex, Preds>> pl, double md)
        : events(move(ev)), activities(move(am)), precedence_list(move(pl)), max_delay(md) {}
};

// ─── simulation result ────────────────────────────────────────────────
struct SimResult {
    vector<double> realized;
    vector<double> delays;
    vector<int> cause;
};

// ─── distributions ────────────────────────────────────────────────────
struct ConstantDist {
    double factor;
    ConstantDist(double f = 0.0) : factor(f) {}
    double sample(mt19937&, double d) const { return d * factor; }
};

struct ExponentialDist {
    double lam, max_scale;
    exponential_distribution<double> dist;
    ExponentialDist(double l = 1.0, double m = 1.0) : lam(l), max_scale(m), dist(1.0 / l) {}
    double sample(mt19937& rng, double d) const {
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
    double sample(mt19937& rng, double d) const {
        double x;
        do {
            x = dist(rng);
        } while (x > max_scale);
        return x * d;
    }
};

// ─── generic delay generator ──────────────────────────────────────────
using DistVar = variant<ConstantDist, ExponentialDist, GammaDist>;
class GenericDelayGenerator {
   public:
    mt19937 rng_;
    unordered_map<int, DistVar> dist_map_;

    GenericDelayGenerator() : rng_(random_device{}()) {}
    void set_seed(int s) { rng_.seed(s); }
    void add_constant(int t, double f) { dist_map_[t] = ConstantDist{f}; }
    void add_exponential(int t, double l, double m) { dist_map_[t] = ExponentialDist{l, m}; }
    void add_gamma(int t, double k, double s, double m = numeric_limits<double>::infinity()) {
        dist_map_[t] = GammaDist{k, s, m};
    }
};

// ─── simulator ────────────────────────────────────────────────────────
class Simulator {
    SimContext ctx_;
    std::vector<DistVar> dists_;
    std::vector<SimActivity> acts_;
    std::vector<int> act2dist_;
    std::mt19937 rng_;
    std::vector<Preds> preds_by_node_;

    // scratch buffers
    std::vector<double> lower_, sched_, comp_, real_;
    std::vector<int> cause_;

   public:
    Simulator(SimContext c, GenericDelayGenerator gen) : ctx_(std::move(c)), rng_(std::random_device{}()) {
        // 1) build type→index & flatten distributions
        std::unordered_map<int, int> type2idx;
        dists_.reserve(gen.dist_map_.size());
        {
            int idx = 0;
            for (auto& kv : gen.dist_map_) {
                type2idx[kv.first] = idx;
                dists_.push_back(kv.second);
                ++idx;
            }
        }

        // 2) flatten activities + record their dist index
        acts_.reserve(ctx_.activities.size());
        act2dist_.reserve(ctx_.activities.size());
        for (auto& kv : ctx_.activities) {
            acts_.push_back(kv.second);
            auto it = type2idx.find(kv.second.activity_type);
            act2dist_.push_back(it == type2idx.end() ? -1 : it->second);
        }

        // 3) build preds_by_node_ from precedence_list
        int ne = int(ctx_.events.size());
        preds_by_node_.assign(ne, {});
        for (auto& p : ctx_.precedence_list) {
            auto& vec = preds_by_node_[p.first];
            for (auto& pr : p.second) vec.push_back(pr);
        }

        // 4) allocate scratch buffers
        int na = int(acts_.size());
        lower_.resize(ne);
        sched_.resize(ne);
        real_.resize(ne);
        cause_.resize(ne);
        comp_.resize(na);
    }

    SimResult run(int seed) {
        // seed RNG
        rng_.seed(seed);

        int ne = int(lower_.size());
        int na = int(comp_.size());

        // load earliest & actual
        for (int i = 0; i < ne; ++i) {
            lower_[i] = ctx_.events[i].ts.earliest;
            sched_[i] = ctx_.events[i].ts.actual;
        }

        // sample + compound durations
        for (int i = 0; i < na; ++i) {
            const auto& act = acts_[i];
            int di = act2dist_[i];
            double extra = 0.0;
            if (di >= 0) {
                extra = std::visit([&](auto& d) { return d.sample(rng_, act.duration); }, dists_[di]);
            }
            comp_[i] = act.duration + extra;
        }

        // init propagate
        std::copy(lower_.begin(), lower_.end(), real_.begin());
        std::iota(cause_.begin(), cause_.end(), 0);

        // propagate through preds_by_node_
        for (int node = 0; node < ne; ++node) {
            for (auto& pr : preds_by_node_[node]) {
                int pred_idx = pr.first;
                int act_idx = pr.second;
                double t = real_[pred_idx] + comp_[act_idx];
                if (t > real_[node]) {
                    real_[node] = t;
                    cause_[node] = pred_idx;
                }
            }
        }

        return SimResult{real_, comp_, cause_};
    }

    std::vector<SimResult> run_many(const std::vector<int>& seeds) {
        std::vector<SimResult> out;
        out.reserve(seeds.size());
        for (int s : seeds) out.push_back(run(s));
        return out;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
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
        .def_readwrite("activity_type", &SimActivity::activity_type, "Type ID for delay distribution");

    // SimContext
    py::class_<SimContext>(m, "SimContext")
        .def(py::init<std::vector<SimEvent>, std::unordered_map<std::pair<int, int>, SimActivity>,
                      std::vector<std::pair<int, Preds>>, double>(),
             py::arg("events"), py::arg("activities"), py::arg("precedence_list"), py::arg("max_delay"),
             "Wraps a DAG: events, activities map, precedence, max delay")
        .def_readwrite("events", &SimContext::events, "List of SimEvent")
        .def_readwrite("activities", &SimContext::activities, "Map (src_idx,dst_idx)→SimActivity")
        .def_readwrite("precedence_list", &SimContext::precedence_list,
                       "List of (target_idx, [(pred_idx, act_idx),...])")
        .def_readwrite("max_delay", &SimContext::max_delay, "Cap on any injected delay");

    // SimResult
    py::class_<SimResult>(m, "SimResult")
        .def_readonly("realized", &SimResult::realized, "Realized event times after propagation")
        .def_readonly("delays", &SimResult::delays, "Injected delays per activity")
        .def_readonly("cause_event", &SimResult::cause, "Index of predecessor causing each event");

    // GenericDelayGenerator
    py::class_<GenericDelayGenerator>(m, "GenericDelayGenerator")
        .def(py::init<>(), "Create a new delay generator")
        .def("set_seed", &GenericDelayGenerator::set_seed, py::arg("seed"), "Set RNG seed for reproducibility")
        .def("add_constant", &GenericDelayGenerator::add_constant, py::arg("activity_type"), py::arg("factor"),
             "Constant: delay = factor × duration")
        .def("add_exponential", &GenericDelayGenerator::add_exponential, py::arg("activity_type"), py::arg("lambda_"),
             py::arg("max_scale"), "Exponential(λ) with cutoff at max_scale")
        .def("add_gamma", &GenericDelayGenerator::add_gamma, py::arg("activity_type"), py::arg("shape"),
             py::arg("scale"), py::arg("max_scale") = std::numeric_limits<double>::infinity(),
             "Gamma(shape, scale) truncated at max_scale");

    // Simulator
    py::class_<Simulator>(m, "Simulator")
        .def(py::init<SimContext, GenericDelayGenerator>(), py::arg("context"), py::arg("generator"),
             "Construct simulator with context and delay generator")
        .def("run", &Simulator::run, py::arg("seed"), "Run a single simulation (seeded)")
        .def("run_many", &Simulator::run_many, py::arg("seeds"), "Run batch of simulations");
}