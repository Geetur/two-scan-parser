#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <climits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace peak {
static_assert(sizeof(int)==4,"32-bit int input and graph encoding required");
// A contiguous machine-word array. Allocation does not initialize every cell;
// the algorithms assign all entries before use. Zeroing is not hidden in timings.
template<class T> class Buffer {
    T* data_=nullptr; size_t size_=0;
public:
    Buffer()=default;
    explicit Buffer(size_t n){resize(n);}
    Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
    ~Buffer(){std::free(data_);}
    void resize(size_t n){
        if(n==size_) return;
        if(n>SIZE_MAX/sizeof(T)-64) throw std::length_error("buffer too large");
        T* next=nullptr;
        if(n){
            size_t bytes=(n*sizeof(T)+63)&~size_t(63);
            next=static_cast<T*>(std::aligned_alloc(64,bytes));
            if(!next) throw std::bad_alloc();
        }
        std::free(data_); data_=next; size_=n;
    }
    T* data(){return data_;} const T* data()const{return data_;}
    T& operator[](size_t i){return data_[i];}
    const T& operator[](size_t i)const{return data_[i];}
    size_t size()const{return size_;}
};

struct Input {
    std::vector<int> states, L, R, rank;
    int n()const{return static_cast<int>(L.size());}
    void validate()const;
};

struct Graph {
    Buffer<int> parent,left,right;
    int root=-1;
    void resize(int n){parent.resize(n);left.resize(n);right.resize(n);root=-1;}
};

// Backend scalar has the same padded state width and phases as the SIMD path.
// Use scalar links on both paths to isolate state-table SIMD from link SIMD.
enum class Backend { Scalar, SIMD4, SIMD8, SIMD16, Auto };
Backend backend_from_string(const std::string& name);
const char* backend_name(Backend backend);
bool available(Backend backend);

class Scan {
public:
    Buffer<int> pop,next;
    int actual_threads=0;
    Scan(const Input& input, Backend backend=Backend::Auto,
         int block=4096,int top_cutoff=128,bool precompute=false,bool vector_links=true,bool optimized_witness=true);
    void parse(const Input& input,Graph& graph,int threads);
    Backend backend()const{return backend_;}
    int width()const{return width_;}
    size_t scratch_bytes()const;
    int block()const{return block_;}
    bool precompute()const{return precompute_;}
    bool optimized_witness()const{return optimized_witness_;}
    bool vector_links()const{
#if defined(__AVX512F__)
        return vector_links_&&backend_!=Backend::Scalar;
#else
        return false;
#endif
    }
private:
    Backend backend_; int n_,c_,width_,block_,cutoff_;
    bool precompute_,vector_links_,optimized_witness_;
    size_t blocks_,S_;
    Buffer<int> after_,tree_,carry_,transitions_;
    Buffer<uint32_t> witness_masks_;
    template<class Ops,bool Precompute,bool Optimized> void run(const Input&,Graph&,int);
    template<class Ops> void run_single(const Input&,Graph&);
};

class Stack {
    Buffer<uint64_t> stack_;
    Buffer<int> last_;
public:
    Buffer<int> pop,next;
    explicit Stack(int n,bool diagnostics=false);
    void parse(const Input&,Graph&,bool diagnostics=false);
    size_t scratch_bytes()const;
private:
    template<bool Diagnostics> void run(const Input&,Graph&);
};

bool equal_graph(const Graph&,const Graph&);
uint64_t checksum(const Graph&);
} // namespace peak
