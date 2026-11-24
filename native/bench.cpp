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

static std::vector<int> numbers(const std::string& text){
    std::vector<int> out;std::stringstream ss(text);std::string item;
    while(std::getline(ss,item,','))out.push_back(std::stoi(item));
    return out;
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

static void verify(){
    std::mt19937 rng(78212);int cases=0,runs=0;
    std::vector<int> sizes={0,1,2,3,4,7,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,511,512,513,1023,1024,1025,4095,4096,4097};
    auto checked=[&](const Input& in,bool all){
        Graph expected,got;std::vector<int> p,nx;oracle(in,expected,p,nx);got.resize(in.n());
        auto poison=[&](){
            if(in.n()){
                std::fill(got.parent.data(),got.parent.data()+in.n(),INT_MIN);
                std::fill(got.left.data(),got.left.data()+in.n(),INT_MIN);
                std::fill(got.right.data(),got.right.data()+in.n(),INT_MIN);
            }
            got.root=INT_MIN;
        };
        poison();Stack serial(in.n(),true);serial.parse(in,got,true);
        if(!equal_graph(expected,got)||!std::equal(p.begin(),p.end(),serial.pop.data())||!std::equal(nx.begin(),nx.end(),serial.next.data()))throw std::runtime_error("packed stack differs from oracle");
        for(Backend b:{Backend::Scalar,Backend::SIMD4,Backend::SIMD8,Backend::SIMD16}){
            int width=b==Backend::SIMD4?4:b==Backend::SIMD8?8:16;
            if(!available(b)||int(in.states.size())>width)continue;
            for(bool optimized:{false,true})for(int threads:all?std::vector<int>{1,4,8}:std::vector<int>{1}){
                Scan scan(in,b,all?127:256,all?8:128,cases%2==0,true,optimized);
                poison();
                if(in.n()){
                    std::fill(scan.pop.data(),scan.pop.data()+in.n(),INT_MIN);
                    std::fill(scan.next.data(),scan.next.data()+in.n(),INT_MIN);
                }
                scan.parse(in,got,threads);
                if(!equal_graph(expected,got)||!std::equal(p.begin(),p.end(),scan.pop.data())||!std::equal(nx.begin(),nx.end(),scan.next.data()))
                    throw std::runtime_error("scan differs: n="+std::to_string(in.n())+" backend="+backend_name(b));
                ++runs;
            }
        }++cases;
    };
    for(int c:{1,2,3,4,5,8,9,16})for(int n:sizes){
        Input in;for(int k=0;k<c;++k)in.states.push_back(k*7-50);
        in.L.resize(n);in.R.resize(n);in.rank.resize(n);
        for(int i=0;i<n;++i){in.rank[i]=rng()%c;in.R[i]=in.states[in.rank[i]];in.L[i]=int(rng()%130)-60;}
        in.validate();checked(in,true);
    }
    // All 9^n numerical streams for n<=4 include ties and blockers.
    for(int n=0;n<=4;++n){
        int total=1;for(int k=0;k<n;++k)total*=9;
        for(int mask=0;mask<total;++mask){
            Input in;in.states={0,1,2};int code=mask;
            for(int k=0;k<n;++k){int digit=code%9;code/=9;in.L.push_back(digit/3);in.R.push_back(digit%3);in.rank.push_back(digit%3);}
            checked(in,false);
        }
    }
    for(int k=0;k<500;++k){
        int c=1+rng()%16,n=rng()%1000;Input in;
        for(int j=0;j<c;++j)in.states.push_back(j*5-25);
        for(int j=0;j<n;++j){int r=rng()%c;in.L.push_back(int(rng()%100)-40);in.R.push_back(in.states[r]);in.rank.push_back(r);}
        checked(in,k%50==0);
    }
    for(auto f:{"sum","power","alternating","statements","mixed","random8"})for(int n:{10000,65536})checked(generate(f,n,true),true);
    std::cout<<"PASS "<<cases<<" input groups; "<<runs<<" scalar/SIMD/thread configurations; all pop, next, parent, left, right and roots match independent oracle.\n";
}

struct Variant {
    std::string name;int threads;bool stack;
    std::unique_ptr<Scan> scan;
    std::unique_ptr<Stack> serial;
    std::unique_ptr<Graph> out;
    std::vector<double> times;
    std::vector<double> cpu_times;
};
static double median(std::vector<double> x){std::sort(x.begin(),x.end());return x.size()%2?x[x.size()/2]:(x[x.size()/2-1]+x[x.size()/2])*0.5;}
static void benchmark(const std::vector<int>& sizes,const std::vector<int>& teams,int repeats,
                      bool fixed,int block,int cutoff,Backend requested,bool allocation,bool precompute,bool vector_links,
                      const std::vector<std::string>& families,const std::string& csv,int inner){
    std::ofstream file(csv);if(!file)throw std::runtime_error("cannot write CSV");
    const char* header="family,n,C,backend,threads,actual_threads,mode,median_ms,min_ms,max_ms,million_ops_s,speedup_vs_stack,scratch_bytes,checksum,width,block,cutoff,precomputed,vector_links,optimized_witness,inner,median_cpu_ms,cpu_over_wall,samples_ms";
    file<<header<<'\n';std::cout<<header<<'\n';
    std::mt19937 random(9234);
    for(auto family:families)for(int n:sizes){
        if(family=="statements"&&n%2==0)--n;
        Input in=generate(family,n,fixed);Graph expected;std::vector<int> p,nx;oracle(in,expected,p,nx);
        std::vector<Variant> variants;
        Variant serial;serial.name="packed_stack";serial.threads=1;serial.stack=true;
        serial.serial=std::make_unique<Stack>(n);serial.out=std::make_unique<Graph>();serial.out->resize(n);variants.push_back(std::move(serial));
        for(bool optimized:{false,true})for(int threads:teams){
            Variant v;v.threads=threads;v.stack=false;v.scan=std::make_unique<Scan>(in,requested,block,cutoff,precompute,vector_links,optimized);
            v.name=std::string(backend_name(v.scan->backend()))+(optimized?"_masks":"_reference");v.out=std::make_unique<Graph>();v.out->resize(n);variants.push_back(std::move(v));
        }
        auto call=[&](Variant& v){
            if(v.stack)v.serial->parse(in,*v.out);
            else v.scan->parse(in,*v.out,v.threads);
        };
        // Pre-fault and warm every workspace; checks are outside all timers.
        for(auto& v:variants){call(v);call(v);if(!equal_graph(expected,*v.out))throw std::runtime_error("benchmark warmup mismatch");}
        std::vector<int> order(variants.size());std::iota(order.begin(),order.end(),0);
        for(int rep=0;rep<repeats;++rep){
            std::shuffle(order.begin(),order.end(),random);
            for(int id:order){auto& v=variants[id];
                auto cpu_start=std::clock();
                auto start=Clock::now();
                if(allocation){
                    Graph fresh;fresh.resize(n);
                    if(v.stack){Stack work(n);work.parse(in,fresh);}
                    else{Scan work(in,v.scan->backend(),block,cutoff,precompute,vector_links,v.scan->optimized_witness());work.parse(in,fresh,v.threads);}
                    auto stop=Clock::now();v.times.push_back(std::chrono::duration<double,std::milli>(stop-start).count());
                    v.cpu_times.push_back(1000.0*(std::clock()-cpu_start)/CLOCKS_PER_SEC);
                    if(!equal_graph(expected,fresh))throw std::runtime_error("allocation mode mismatch");
                }else{
                    for(int iteration=0;iteration<inner;++iteration)call(v);
                    auto stop=Clock::now();
                    v.times.push_back(std::chrono::duration<double,std::milli>(stop-start).count()/inner);
                    v.cpu_times.push_back(1000.0*(std::clock()-cpu_start)/CLOCKS_PER_SEC/inner);
                    if(!equal_graph(expected,*v.out))throw std::runtime_error("benchmark mismatch");
                }
            }
        }
        double baseline=median(variants[0].times);
        for(auto& v:variants){
            double ms=median(v.times);std::ostringstream line;
            line<<family<<','<<n<<','<<in.states.size()<<','<<v.name<<','<<v.threads<<','<<(v.stack?1:v.scan->actual_threads)<<','<<(allocation?"allocate_parse":"reuse_parse")<<','<<std::setprecision(9)<<ms<<','<<*std::min_element(v.times.begin(),v.times.end())<<','<<*std::max_element(v.times.begin(),v.times.end())<<','<<n/(ms*1000)<<','<<baseline/ms<<','<<(v.stack?v.serial->scratch_bytes():v.scan->scratch_bytes())<<','<<checksum(*v.out)<<','<<(v.stack?0:v.scan->width())<<','<<(v.stack?0:block)<<','<<(v.stack?0:cutoff)<<','<<(!v.stack&&precompute)<<','<<(!v.stack&&v.scan->vector_links())<<','<<(!v.stack&&v.scan->optimized_witness())<<','<<inner<<','<<median(v.cpu_times)<<','<<median(v.cpu_times)/ms<<",\"";
            for(size_t k=0;k<v.times.size();++k){if(k)line<<';';line<<v.times[k];}line<<'"';
            file<<line.str()<<'\n';file.flush();std::cout<<line.str()<<std::endl;
        }
    }
}

int main(int argc,char** argv){
    try{
        std::vector<int> sizes={65536,1048576,4194304},teams={1,2,4,8};int repeats=9,block=4096,cutoff=128,inner=1;
        bool fixed=true,allocation=false,check=false,precompute=false,vector_links=true;Backend backend=Backend::Auto;std::string csv="results/benchmark.csv";
        std::vector<std::string> families={"sum","power","alternating","statements","mixed","random8"};
        for(int k=1;k<argc;++k){std::string opt=argv[k];auto value=[&](){if(++k>=argc)throw std::invalid_argument("missing option value");return std::string(argv[k]);};
            if(opt=="--verify")check=true;
            else if(opt=="--sizes")sizes=numbers(value());
            else if(opt=="--threads")teams=numbers(value());
            else if(opt=="--repeats")repeats=std::stoi(value());
            else if(opt=="--inner")inner=std::stoi(value());
            else if(opt=="--block")block=std::stoi(value());
            else if(opt=="--cutoff")cutoff=std::stoi(value());
            else if(opt=="--backend")backend=backend_from_string(value());
            else if(opt=="--used-states")fixed=false;
            else if(opt=="--allocation")allocation=true;
            else if(opt=="--inline")precompute=false;
            else if(opt=="--precompute")precompute=true;
            else if(opt=="--scalar-links")vector_links=false;
            else if(opt=="--families"){
                std::stringstream input(value());std::string item;families.clear();
                while(std::getline(input,item,','))families.push_back(item);
            }
            else if(opt=="--csv")csv=value();
            else throw std::invalid_argument("unknown option: "+opt);
        }
        omp_set_dynamic(0);
        if(check){verify();return 0;}
        if(repeats<1||inner<1||(allocation&&inner!=1)||sizes.empty()||teams.empty())throw std::invalid_argument("empty benchmark settings");
        for(int n:sizes)if(n<1||n>INT_MAX-2)throw std::invalid_argument("invalid size");
        for(int p:teams)if(p<1||p>256)throw std::invalid_argument("invalid team");
        benchmark(sizes,teams,repeats,fixed,block,cutoff,backend,allocation,precompute,vector_links,families,csv,inner);
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
