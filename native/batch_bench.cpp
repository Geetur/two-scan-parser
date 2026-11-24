#include "parser.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <functional>
#include <omp.h>

using namespace peak;
using Clock=std::chrono::steady_clock;
static constexpr int INF=1000000000;

static Input generate(const std::string& family,int n,bool fixed,unsigned seed=93211){
    Input in;in.L.resize(n);in.R.resize(n);in.rank.resize(n);
    std::mt19937 rng(seed);
    for(int i=0;i<n;++i){
        if(family=="sum"){in.L[i]=30;in.R[i]=31;}
        else if(family=="power"){in.L[i]=51;in.R[i]=50;}
        else if(family=="alternating"){
            in.L[i]=i%2?51:INF;in.R[i]=i%2?50:45;
        }else if(family=="statements"){
            in.L[i]=i%2?1:10;in.R[i]=i%2?2:INF;
        }else if(family=="mixed"){
            // Conventional valid arithmetic: +,-,*,/,^ with source operands
            // between operators. Neither parser receives the family label.
            static constexpr int left[]={30,30,40,40,51};
            static constexpr int right[]={31,31,41,41,50};
            unsigned op=rng()%5;in.L[i]=left[op];in.R[i]=right[op];
        }else if(family=="random8"){
            static constexpr int powers[]={2,5,20,31,41,45,50,INF};
            in.L[i]=int(rng()%60);in.R[i]=powers[rng()%8];
        }else throw std::invalid_argument("unknown family");
    }
    if(fixed||family=="random8")in.states={2,5,20,31,41,45,50,INF};
    else{in.states=in.R;std::sort(in.states.begin(),in.states.end());in.states.erase(std::unique(in.states.begin(),in.states.end()),in.states.end());}
    if(in.states.empty())in.states={0};
    for(int i=0;i<n;++i)in.rank[i]=int(std::lower_bound(in.states.begin(),in.states.end(),in.R[i])-in.states.begin());
    in.validate();return in;
}

// An independent two-stack oracle, separate from the packed-stack baseline.
static void oracle(const Input& in,Graph& g,std::vector<int>& pop,std::vector<int>& next){
    int n=in.n();g.resize(n);pop.assign(n,n);next.assign(n,-1);
    if(n)std::fill(g.parent.data(),g.parent.data()+n,-1);
    std::vector<int> operators,values={-1};
    auto fire=[&](int event){
        int i=operators.back();operators.pop_back();
        int b=values.back();values.pop_back();int a=values.back();values.pop_back();
        pop[i]=event;g.left[i]=a;g.right[i]=b;
        if(a>=0)g.parent[a]=i;
        if(b>=0)g.parent[b]=i;
        values.push_back(i);
    };
    for(int j=0;j<n;++j){
        while(!operators.empty()&&in.R[operators.back()]>in.L[j])fire(j);
        operators.push_back(j);values.push_back(-(j+2));
    }
    while(!operators.empty())fire(n);
    g.root=n?values[0]:-1;
    std::vector<int> last(n+1,-1);
    for(int i=n-1;i>=0;--i){next[i]=last[pop[i]];last[pop[i]]=i;}
}


static double median(std::vector<double> values){
    std::sort(values.begin(),values.end());return values[values.size()/2];
}
struct Case {
    Input input;
    Graph expected,stack_out,reference_out,masks_out;
};
int main(int argc,char** argv){
    try{
        int groups=1024,repeats=11;
        std::string filename="results/batches.csv";
        for(int i=1;i<argc;++i){
            std::string option=argv[i];
            if(++i==argc)throw std::invalid_argument("missing value");
            if(option=="--groups")groups=std::stoi(argv[i]);
            else if(option=="--repeats")repeats=std::stoi(argv[i]);
            else if(option=="--csv")filename=argv[i];
            else throw std::invalid_argument("unknown option");
        }
        if(groups<1||repeats<1)throw std::invalid_argument("invalid dimensions");
        omp_set_dynamic(0);
        std::ofstream csv(filename);
        csv<<"family,group_size,groups,total_operators,variant,threads,median_ms,ns_per_operator,stack_same_threads_speedup,samples_ms\n";
        std::mt19937 rng(94382);
        for(std::string family:{"mixed","random8"})for(int n:{16,64,256,1024}){
            std::vector<std::unique_ptr<Case>> cases;
            for(int i=0;i<groups;++i){
                auto c=std::make_unique<Case>();
                c->input=generate(family,n,true,93211u+7919u*unsigned(i));
                std::vector<int> pop,next;oracle(c->input,c->expected,pop,next);
                c->stack_out.resize(n);c->reference_out.resize(n);c->masks_out.resize(n);
                cases.push_back(std::move(c));
            }
            // A bounded workspace pool is reused across independent groups.
            std::vector<std::unique_ptr<Stack>> stacks;
            std::vector<std::unique_ptr<Scan>> reference,masks;
            for(int i=0;i<8;++i){
                stacks.push_back(std::make_unique<Stack>(n));
                reference.push_back(std::make_unique<Scan>(cases[0]->input,Backend::Auto,4096,128,false,true,false));
                masks.push_back(std::make_unique<Scan>(cases[0]->input,Backend::Auto,4096,128,false,true,true));
            }
            struct Variant {int mode,threads;std::vector<double> samples;};
            std::vector<Variant> variants;
            for(int team:{1,4,8})for(int mode:{0,1,2})variants.push_back({mode,team,{}});
            auto call=[&](Variant& v){
                #pragma omp parallel for num_threads(v.threads) schedule(static)
                for(int i=0;i<groups;++i){
                    auto& c=*cases[i];int worker=omp_get_thread_num();
                    if(v.mode==0)stacks[worker]->parse(c.input,c.stack_out);
                    else if(v.mode==1)reference[worker]->parse(c.input,c.reference_out,1);
                    else masks[worker]->parse(c.input,c.masks_out,1);
                }
            };
            auto check=[&](int mode){
                for(auto& c:cases){
                    const Graph& graph=mode==0?c->stack_out:mode==1?c->reference_out:c->masks_out;
                    if(!equal_graph(graph,c->expected))throw std::runtime_error("batch graph mismatch");
                }
            };
            for(auto& v:variants){call(v);call(v);check(v.mode);}
            std::vector<int> order(variants.size());std::iota(order.begin(),order.end(),0);
            for(int repeat=0;repeat<repeats;++repeat){
                std::shuffle(order.begin(),order.end(),rng);
                for(int i:order){auto& v=variants[i];auto start=Clock::now();call(v);
                    auto stop=Clock::now();v.samples.push_back(std::chrono::duration<double,std::milli>(stop-start).count());
                    check(v.mode);
                }
            }
            for(auto& v:variants){
                double time=median(v.samples),baseline=0;
                for(auto& other:variants)if(other.mode==0&&other.threads==v.threads)baseline=median(other.samples);
                const char* name=v.mode==0?"packed_stack":v.mode==1?"simd_reference":"simd_masks";
                csv<<family<<','<<n<<','<<groups<<','<<size_t(n)*groups<<','<<name<<','<<v.threads<<','<<std::setprecision(9)<<time<<','<<time*1e6/(size_t(n)*groups)<<','<<baseline/time<<",\"";
                for(size_t i=0;i<v.samples.size();++i){if(i)csv<<';';csv<<v.samples[i];}
                csv<<"\"\n";csv.flush();
                std::cout<<family<<" n="<<n<<" "<<name<<" team="<<v.threads<<" ms="<<time<<" vs_stack="<<baseline/time<<std::endl;
            }
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
