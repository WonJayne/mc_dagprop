// mc_dagprop/_core.cpp

#include <algorithm>
#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <variant>

namespace py = pybind11;

// ------------------------- Custom Hash for pair -------------------------
namespace std {
    template <typename T1, typename T2>
    struct hash<std::pair<T1, T2>> {
        size_t operator()(const std::pair<T1, T2>& p) const noexcept {
            return std::hash<T1>{}(p.first) ^ (std::hash<T2>{}(p.second) << 1);
        }
    };
}

// ------------------------- Data Types -------------------------
// renamed from SimEvent ? EventTimestamp
struct EventTimestamp {
    double earliest;
    double latest;
    double actual;
};

// an Event in the DAG
struct SimEvent {
    std::string    node_id;
    EventTimestamp timestamp;
};

// an Activity (edge) in the DAG
struct SimActivity {
    double duration;
    int    activity_type;
};

using NodeIndex  = int;
using Precedence = std::vector<std::pair<NodeIndex, NodeIndex>>; // (pred_event_idx, activity_idx)

// ------------------------- Simulation Context -------------------------
struct SimContext {
    std::vector<SimEvent>                                events;
    std::unordered_map<std::pair<NodeIndex, NodeIndex>, SimActivity> activities;
    std::vector<std::pair<NodeIndex, Precedence>>        precedence_list;
    double                                                max_delay;

    SimContext(
        std::vector<SimEvent>                                ev,
        std::unordered_map<std::pair<NodeIndex, NodeIndex>, SimActivity> am,
        std::vector<std::pair<NodeIndex, Precedence>>        pl,
        double                                                md
    )
      : events(std::move(ev))
      , activities(std::move(am))
      , precedence_list(std::move(pl))
      , max_delay(md)
    {}
};

// ------------------------- Simulation Result -------------------------
struct SimResult {
    std::vector<double> realized;
    std::vector<double> delays;
    std::vector<int>    cause_event;
};

// ------------------------- Delay Distributions -------------------------
struct ConstantDist {
    double factor;
    ConstantDist(): factor(0.0) {}
    ConstantDist(double f): factor(f) {}
    double sample(std::mt19937&, double d) const {
        return d * factor;
    }
};

struct ExponentialDist {
    double lambda, max_scale;
    std::exponential_distribution<double> dist;
    ExponentialDist(): lambda(1.0), max_scale(1.0), dist(1.0) {}
    ExponentialDist(double lam, double mx)
      : lambda(lam), max_scale(mx), dist(1.0/lam)
    {}
    double sample(std::mt19937& rng, double d) const {
        double x;
        do {
            x = dist(rng);
        } while (x > max_scale);
        return x * d;
    }
};

// **new**: Gamma distribution (shape?scale parametrisation)
struct GammaDist {
    double shape, scale;
    std::gamma_distribution<double> dist;
    GammaDist(): shape(1.0), scale(1.0), dist(1.0, 1.0) {}
    GammaDist(double k, double a)
      : shape(k), scale(a), dist(k, a)
    {}
    double sample(std::mt19937& rng, double d) const {
        return dist(rng) * d;
    }
};

// ------------------------- Generic Delay Generator -------------------------
class GenericDelayGenerator {
    std::mt19937 rng_;
    using DistVar = std::variant<ConstantDist, ExponentialDist, GammaDist>;
    std::unordered_map<int, DistVar> dist_map_;

public:
    GenericDelayGenerator()
      : rng_(std::random_device{}())
    {}

    void set_seed(int seed) {
        rng_.seed(seed);
    }

    void add_constant(int activity_type, double factor) {
        dist_map_.insert_or_assign(activity_type, ConstantDist{factor});
    }

    void add_exponential(int activity_type, double lambda, double max_scale) {
        dist_map_.insert_or_assign(activity_type, ExponentialDist{lambda, max_scale});
    }

    /// New: gamma(shape, scale)
    void add_gamma(int activity_type, double shape, double scale) {
        dist_map_.insert_or_assign(activity_type, GammaDist{shape, scale});
    }

    double get_delay(const SimActivity& act) {
        auto it = dist_map_.find(act.activity_type);
        if (it == dist_map_.end())
            return 0.0;
        // dispatch to the appropriate sample()
        return std::visit(
            [&](auto& d){ return d.sample(rng_, act.duration); },
            it->second
        );
    }
};

// ------------------------- Simulator -------------------------
class Simulator {
    SimContext            ctx_;
    GenericDelayGenerator gen_;
    std::vector<SimActivity> acts_;
    std::vector<bool>        is_affected_;
    std::vector<double>      base_dur_;

public:
    Simulator(SimContext c, GenericDelayGenerator g)
      : ctx_(std::move(c))
      , gen_(std::move(g))
    {
        int n = int(ctx_.activities.size());
        acts_.resize(n);
        is_affected_.assign(n, false);
        base_dur_.assign(n, 0.0);

        int idx = 0;
        for (auto& kv : ctx_.activities) {
            acts_[idx]       = kv.second;
            is_affected_[idx] = (kv.second.activity_type == 1 && kv.second.duration >= 0.001);
            base_dur_[idx]    = kv.second.duration;
            ++idx;
        }
    }

    SimResult run(int seed) {
        gen_.set_seed(seed);
        int ne = int(ctx_.events.size());
        int na = int(acts_.size());

        std::vector<double> lower(ne), scheduled(ne);
        for (auto i = 0; i < ne; ++i) {
            lower[i]     = ctx_.events[i].timestamp.earliest;
            scheduled[i] = ctx_.events[i].timestamp.actual;
        }

        std::vector<double> compounded(na);
        for (auto i = 0; i < na; ++i) {
            compounded[i] = base_dur_[i] + (is_affected_[i] ? gen_.get_delay(acts_[i]) : 0.0);
        }

        std::vector<double> realized = lower;
        std::vector<int>    cause_event(ne);
        std::iota(cause_event.begin(), cause_event.end(), 0);

        for (auto& p : ctx_.precedence_list) {
            int node = p.first;
            auto& preds = p.second;
            if (preds.size() == 1) {
                int pi = preds[0].first, ai = preds[0].second;
                double t = realized[pi] + compounded[ai];
                if (t > realized[node]) {
                    realized[node]    = t;
                    cause_event[node] = pi;
                }
            } else if (!preds.empty()) {
                double mx = -1e9;
                int bi = 0;
                for (int i = 0; i < int(preds.size()); ++i) {
                    int pi = preds[i].first, ai = preds[i].second;
                    double t = realized[pi] + compounded[ai];
                    if (t > mx) { mx = t; bi = i; }
                }
                if (mx > realized[node]) {
                    realized[node]    = mx;
                    cause_event[node] = preds[bi].first;
                }
            }
        }

        return SimResult{std::move(realized), std::move(compounded), std::move(cause_event)};
    }

    std::vector<SimResult> run_many(const std::vector<int>& seeds) {
        std::vector<SimResult> out;
        out.reserve(seeds.size());
        for (int s : seeds) out.push_back(run(s));
        return out;
    }
};

// ------------------------- Pybind11 Exports -------------------------
PYBIND11_MODULE(_core, m) {
    m.doc() = "Core Monte-Carlo DAG-propagation simulator";

    // EventTimestamp
    py::class_<EventTimestamp>(m, "EventTimestamp")
        .def(py::init<double, double, double>(),
             py::arg("earliest"), py::arg("latest"), py::arg("actual"))
        .def_readwrite("earliest", &EventTimestamp::earliest)
        .def_readwrite("latest",   &EventTimestamp::latest)
        .def_readwrite("actual",   &EventTimestamp::actual)
        ;

    // SimEvent
    py::class_<SimEvent>(m, "SimEvent")
        .def(py::init<std::string, EventTimestamp>(),
             py::arg("node_id"), py::arg("timestamp"))
        .def_readwrite("node_id",   &SimEvent::node_id)
        .def_readwrite("timestamp", &SimEvent::timestamp)
        ;

    // SimActivity
    py::class_<SimActivity>(m, "SimActivity")
        .def(py::init<double, int>(),
             py::arg("duration"), py::arg("activity_type"))
        .def_readwrite("duration",      &SimActivity::duration)
        .def_readwrite("activity_type", &SimActivity::activity_type)
        ;

    // SimContext
    py::class_<SimContext>(m, "SimContext")
        .def(py::init<
            std::vector<SimEvent>,
            std::unordered_map<std::pair<int,int>,SimActivity>,
            std::vector<std::pair<int,Precedence>>,
            double>(),
             py::arg("events"),
             py::arg("activities"),
             py::arg("precedence_list"),
             py::arg("max_delay"))
        .def_readwrite("events",          &SimContext::events)
        .def_readwrite("activities",      &SimContext::activities)
        .def_readwrite("precedence_list", &SimContext::precedence_list)
        .def_readwrite("max_delay",       &SimContext::max_delay)
        ;

    // SimResult
    py::class_<SimResult>(m, "SimResult")
        .def_readonly("realized",    &SimResult::realized)
        .def_readonly("delays",      &SimResult::delays)
        .def_readonly("cause_event", &SimResult::cause_event)
        ;

    // GenericDelayGenerator
    py::class_<GenericDelayGenerator>(m, "GenericDelayGenerator")
        .def(py::init<>())
        .def("set_seed",        &GenericDelayGenerator::set_seed,        py::arg("seed"))
        .def("add_constant",    &GenericDelayGenerator::add_constant,    py::arg("activity_type"), py::arg("factor"))
        .def("add_exponential", &GenericDelayGenerator::add_exponential, py::arg("activity_type"), py::arg("lambda"), py::arg("max_scale"))
        .def("add_gamma",       &GenericDelayGenerator::add_gamma,       py::arg("activity_type"), py::arg("shape"),  py::arg("scale"))
        ;

    // Simulator
    py::class_<Simulator>(m, "Simulator")
        .def(py::init<SimContext, GenericDelayGenerator>(),
             py::arg("context"), py::arg("generator"))
        .def("run",       &Simulator::run,      py::arg("seed"))
        .def("run_many",  &Simulator::run_many, py::arg("seeds"))
        ;
}
