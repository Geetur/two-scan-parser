#include "parser.hpp"
#include <array>
#include <cstring>
#include <omp.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace peak {
void Input::validate()const {
    if(L.size()>size_t(INT_MAX-2)||R.size()!=L.size()||rank.size()!=L.size())
        throw std::invalid_argument("input lengths or index limit");
    if(states.empty()||states.size()>16) throw std::invalid_argument("need 1..16 states");
    for(size_t k=1;k<states.size();++k)
        if(states[k]<=states[k-1])throw std::invalid_argument("states not ordered");
    for(size_t i=0;i<L.size();++i)
        if(rank[i]<0||size_t(rank[i])>=states.size()||states[rank[i]]!=R[i])
            throw std::invalid_argument("inconsistent encoded right power");
}

Backend backend_from_string(const std::string& s){
    if(s=="auto")return Backend::Auto;
    if(s=="scalar")return Backend::Scalar;
    if(s=="simd4")return Backend::SIMD4;
    if(s=="simd8")return Backend::SIMD8;
    if(s=="simd16")return Backend::SIMD16;
    throw std::invalid_argument("unknown backend: "+s);
}
const char* backend_name(Backend b){
    switch(b){case Backend::Scalar:return "scalar";case Backend::SIMD4:return "simd4";
    case Backend::SIMD8:return "simd8";case Backend::SIMD16:return "simd16";
    default:return "auto";}
}
bool available(Backend b){
    if(b==Backend::Auto||b==Backend::Scalar)return true;
#if defined(__AVX__)
    if(b==Backend::SIMD4)return __builtin_cpu_supports("avx")&&__builtin_cpu_supports("sse4.1");
#endif
#if defined(__AVX2__)
    if(b==Backend::SIMD8)return __builtin_cpu_supports("avx2");
#endif
#if defined(__AVX512F__)
    if(b==Backend::SIMD16)return __builtin_cpu_supports("avx512f");
#endif
    return false;
}

// Negative codes ~j are absorbing terminals; nonnegative codes are live
// state indices. A register permutation performs all indexed table lookups.
template<int Width> struct Scalar {
    static constexpr int W=Width;
    using V=std::array<int,W>;
    static V load(const int* p){V v;std::copy(p,p+W,v.begin());return v;}
    static void store(int* p,const V& v){std::copy(v.begin(),v.end(),p);}
    static V identity(){V v;for(int k=0;k<W;++k)v[k]=k;return v;}
    static V compose(const V& a,const V& b){V v;for(int k=0;k<W;++k)v[k]=a[k]<0?a[k]:b[a[k]];return v;}
    static V first(const V& current,const V& states,int left,int rank,int j){
        V v;for(int k=0;k<W;++k)v[k]=states[k]>left?~j:current[std::min(k,rank)];return v;
    }
    static V second(const V& current,const V& states,int left,int rank,int j,int n,const V& after,int pop){
        V v;for(int k=0;k<W;++k){int x=std::min(k,rank),hit=after[x]<0?~after[x]:n;
            v[k]=states[k]>left?~n:(hit==pop?~j:current[x]);}return v;
    }
    static uint32_t prepare_second(const V& after,const V& states,int left,int rank,int jpop,int n){
        uint32_t witness=0,stop=0;
        for(int k=0;k<W;++k){
            int hit=after[k]<0?~after[k]:n;
            if(k>=rank||hit==jpop)witness|=1u<<k;
            if(states[k]>left)stop|=1u<<k;
        }
        return witness|(stop<<16);
    }
    static V second_mask(const V& current,uint32_t masks,int j,int n){
        V result;
        for(int k=0;k<W;++k)
            result[k]=(masks&(1u<<(k+16)))?~n:(masks&(1u<<k))?~j:current[k];
        return result;
    }
    static V diagonal(const V& a,const V& b){
        V result;for(int k=0;k<W;++k)result[k]=a[k]<0?a[k]:b[k];return result;
    }

};

#if defined(__AVX__)
struct SIMD4 {
    static constexpr int W=4;using V=__m128i;
    static V load(const int* p){return _mm_loadu_si128(reinterpret_cast<const V*>(p));}
    static void store(int* p,V v){_mm_storeu_si128(reinterpret_cast<V*>(p),v);}
    static V identity(){return _mm_setr_epi32(0,1,2,3);}
    static V pick(V v,V idx){return _mm_castps_si128(_mm_permutevar_ps(_mm_castsi128_ps(v),idx));}
    static V select(V mask,V yes,V no){return _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(no),_mm_castsi128_ps(yes),_mm_castsi128_ps(mask)));}
    static V compose(V a,V b){return select(a,a,pick(b,a));}
    static V first(V current,V states,int left,int rank,int j){
        V idx=_mm_min_epi32(identity(),_mm_set1_epi32(rank));
        return select(_mm_cmpgt_epi32(states,_mm_set1_epi32(left)),_mm_set1_epi32(~j),pick(current,idx));
    }
    static V second(V current,V states,int left,int rank,int j,int n,V after,int pop){
        V idx=_mm_min_epi32(identity(),_mm_set1_epi32(rank)),code=pick(after,idx);
        V hit=select(code,code,_mm_set1_epi32(~n));
        V result=select(_mm_cmpeq_epi32(hit,_mm_set1_epi32(~pop)),_mm_set1_epi32(~j),pick(current,idx));
        return select(_mm_cmpgt_epi32(states,_mm_set1_epi32(left)),_mm_set1_epi32(~n),result);
    }
    static uint32_t prepare_second(V after,V states,int left,int rank,int jpop,int n){
        V hit=select(after,after,_mm_set1_epi32(~n));
        uint32_t witness=_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(hit,_mm_set1_epi32(~jpop))));
        witness|=(15u<<rank)&15u;
        uint32_t stop=_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(states,_mm_set1_epi32(left))));
        return witness|(stop<<16);
    }
    static V second_mask(V current,uint32_t masks,int j,int n){
#if defined(__AVX512VL__)
        V result=_mm_mask_mov_epi32(current,(__mmask8)masks,_mm_set1_epi32(~j));
        return _mm_mask_mov_epi32(result,(__mmask8)(masks>>16),_mm_set1_epi32(~n));
#else
        const V bits=_mm_setr_epi32(1,2,4,8);
        V witness=_mm_cmpgt_epi32(_mm_and_si128(bits,_mm_set1_epi32(masks)),_mm_setzero_si128());
        V stop=_mm_cmpgt_epi32(_mm_and_si128(bits,_mm_set1_epi32(masks>>16)),_mm_setzero_si128());
        return select(stop,_mm_set1_epi32(~n),select(witness,_mm_set1_epi32(~j),current));
#endif
    }
    static V diagonal(V a,V b){return select(a,a,b);}

};
#endif
#if defined(__AVX2__)
struct SIMD8 {
    static constexpr int W=8;using V=__m256i;
    static V load(const int* p){return _mm256_loadu_si256(reinterpret_cast<const V*>(p));}
    static void store(int* p,V v){_mm256_storeu_si256(reinterpret_cast<V*>(p),v);}
    static V identity(){return _mm256_setr_epi32(0,1,2,3,4,5,6,7);}
    static V pick(V v,V idx){return _mm256_permutevar8x32_epi32(v,idx);}
    static V select(V mask,V yes,V no){return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(no),_mm256_castsi256_ps(yes),_mm256_castsi256_ps(mask)));}
    static V compose(V a,V b){
#if defined(__AVX512VL__)
        return _mm256_mask_permutexvar_epi32(a,_mm256_cmp_epi32_mask(a,_mm256_setzero_si256(),_MM_CMPINT_NLT),a,b);
#else
        return select(a,a,pick(b,a));
#endif
    }
    static V first(V current,V states,int left,int rank,int j){
        V idx=_mm256_min_epi32(identity(),_mm256_set1_epi32(rank));
        return select(_mm256_cmpgt_epi32(states,_mm256_set1_epi32(left)),_mm256_set1_epi32(~j),pick(current,idx));
    }
    static V second(V current,V states,int left,int rank,int j,int n,V after,int pop){
        V idx=_mm256_min_epi32(identity(),_mm256_set1_epi32(rank)),code=pick(after,idx);
        V hit=select(code,code,_mm256_set1_epi32(~n));
        V result=select(_mm256_cmpeq_epi32(hit,_mm256_set1_epi32(~pop)),_mm256_set1_epi32(~j),pick(current,idx));
        return select(_mm256_cmpgt_epi32(states,_mm256_set1_epi32(left)),_mm256_set1_epi32(~n),result);
    }
    static uint32_t prepare_second(V after,V states,int left,int rank,int jpop,int n){
        V hit=select(after,after,_mm256_set1_epi32(~n));
        uint32_t witness=_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(hit,_mm256_set1_epi32(~jpop))));
        witness|=(255u<<rank)&255u;
        uint32_t stop=_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(states,_mm256_set1_epi32(left))));
        return witness|(stop<<16);
    }
    static V second_mask(V current,uint32_t masks,int j,int n){
#if defined(__AVX512VL__)
        V result=_mm256_mask_mov_epi32(current,(__mmask8)masks,_mm256_set1_epi32(~j));
        return _mm256_mask_mov_epi32(result,(__mmask8)(masks>>16),_mm256_set1_epi32(~n));
#else
        const V bits=_mm256_setr_epi32(1,2,4,8,16,32,64,128);
        V witness=_mm256_cmpgt_epi32(_mm256_and_si256(bits,_mm256_set1_epi32(masks)),_mm256_setzero_si256());
        V stop=_mm256_cmpgt_epi32(_mm256_and_si256(bits,_mm256_set1_epi32(masks>>16)),_mm256_setzero_si256());
        return select(stop,_mm256_set1_epi32(~n),select(witness,_mm256_set1_epi32(~j),current));
#endif
    }
    static V diagonal(V a,V b){return select(a,a,b);}

};
#endif
#if defined(__AVX512F__)
struct SIMD16 {
    static constexpr int W=16;using V=__m512i;
    static V load(const int* p){return _mm512_loadu_si512(p);}
    static void store(int* p,V v){_mm512_storeu_si512(p,v);}
    static V identity(){return _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);}
    static V pick(V v,V idx){return _mm512_permutexvar_epi32(idx,v);}
    static V compose(V a,V b){return _mm512_mask_mov_epi32(pick(b,a),_mm512_cmp_epi32_mask(a,_mm512_setzero_si512(),_MM_CMPINT_LT),a);}
    static V first(V current,V states,int left,int rank,int j){
        V idx=_mm512_min_epi32(identity(),_mm512_set1_epi32(rank));
        return _mm512_mask_mov_epi32(pick(current,idx),_mm512_cmp_epi32_mask(states,_mm512_set1_epi32(left),_MM_CMPINT_GT),_mm512_set1_epi32(~j));
    }
    static V second(V current,V states,int left,int rank,int j,int n,V after,int pop){
        V idx=_mm512_min_epi32(identity(),_mm512_set1_epi32(rank)),code=pick(after,idx);
        V hit=_mm512_mask_mov_epi32(_mm512_set1_epi32(~n),_mm512_cmp_epi32_mask(code,_mm512_setzero_si512(),_MM_CMPINT_LT),code);
        V result=_mm512_mask_mov_epi32(pick(current,idx),_mm512_cmpeq_epi32_mask(hit,_mm512_set1_epi32(~pop)),_mm512_set1_epi32(~j));
        return _mm512_mask_mov_epi32(result,_mm512_cmp_epi32_mask(states,_mm512_set1_epi32(left),_MM_CMPINT_GT),_mm512_set1_epi32(~n));
    }
    static uint32_t prepare_second(V after,V states,int left,int rank,int jpop,int n){
        V hit=_mm512_mask_mov_epi32(_mm512_set1_epi32(~n),_mm512_cmp_epi32_mask(after,_mm512_setzero_si512(),_MM_CMPINT_LT),after);
        uint32_t witness=_mm512_cmpeq_epi32_mask(hit,_mm512_set1_epi32(~jpop));
        witness|=(65535u<<rank)&65535u;
        uint32_t stop=_mm512_cmp_epi32_mask(states,_mm512_set1_epi32(left),_MM_CMPINT_GT);
        return witness|(stop<<16);
    }
    static V second_mask(V current,uint32_t masks,int j,int n){
        V result=_mm512_mask_mov_epi32(current,(__mmask16)masks,_mm512_set1_epi32(~j));
        return _mm512_mask_mov_epi32(result,(__mmask16)(masks>>16),_mm512_set1_epi32(~n));
    }
    static V diagonal(V a,V b){
        return _mm512_mask_mov_epi32(b,_mm512_cmp_epi32_mask(a,_mm512_setzero_si512(),_MM_CMPINT_LT),a);
    }

};
#endif

Scan::Scan(const Input& in,Backend b,int block,int cutoff,bool precompute,bool vector_links,bool optimized_witness):
    backend_(b),n_(in.n()),c_(int(in.states.size())),block_(block),cutoff_(cutoff),precompute_(precompute),vector_links_(vector_links),optimized_witness_(optimized_witness){
    if(block<1||block>65536||cutoff<1||cutoff>4096||(cutoff&(cutoff-1)))
        throw std::invalid_argument("invalid block / power-of-two top cutoff");
    if(c_<1||c_>16||n_<0)throw std::invalid_argument("invalid dimensions");
    width_=c_<=4?4:c_<=8?8:16;
    if(b==Backend::Auto){
        backend_=width_==4?Backend::SIMD4:width_==8?Backend::SIMD8:Backend::SIMD16;
        if(!available(backend_))backend_=Backend::Scalar;
    }else if(b!=Backend::Scalar){
        width_=b==Backend::SIMD4?4:b==Backend::SIMD8?8:16;
        if(c_>width_||!available(b))throw std::invalid_argument("SIMD backend unavailable / too narrow");
    }
    blocks_=(size_t(n_)+block_-1)/block_; S_=1;while(S_<blocks_)S_*=2;
    pop.resize(n_);next.resize(n_);
    if(optimized_witness_)witness_masks_.resize(n_);else after_.resize(size_t(n_)*width_);
    tree_.resize(2*S_*width_);carry_.resize(2*S_*width_);
    if(precompute_)transitions_.resize(size_t(n_)*width_);
}
size_t Scan::scratch_bytes()const{return sizeof(int)*(pop.size()+next.size()+after_.size()+tree_.size()+carry_.size()+transitions_.size()+witness_masks_.size());}

template<class Ops,bool Precompute,bool Optimized> void Scan::run(const Input& in,Graph& graph,int threads){
    constexpr int W=Ops::W;using V=typename Ops::V;
    alignas(64) int state_values[W];std::fill(state_values,state_values+W,INT_MAX);
    std::copy(in.states.begin(),in.states.end(),state_values);
    alignas(64) const V states=Ops::load(state_values);
    const int n=n_,B=block_;const size_t S=S_,blocks=blocks_;
    int *a=after_.data(),*t=tree_.data(),*v=carry_.data();
    int *tr=transitions_.data();auto* masks=witness_masks_.data();
    int *p=pop.data(),*nx=next.data(),*pa=graph.parent.data(),*le=graph.left.data(),*ri=graph.right.data();
    const int *left=in.L.data(),*rank=in.rank.data();
    int found=-1;
    // Upper tree has at most a fixed number of nodes. One worker handles it
    // to remove many tiny team barriers. The cutoff is independent of n.
    #pragma omp parallel num_threads(threads) reduction(max:found)
    {
        #pragma omp single
        actual_threads=omp_get_num_threads();
        for(int pass=0;pass<2;++pass){
            auto combine=[&](V first,V second){
                if constexpr(Optimized){if(pass==1)return Ops::diagonal(first,second);}
                return Ops::compose(first,second);
            };
            if constexpr(Precompute){
                #pragma omp for schedule(static)
                for(int j=0;j<n;++j){
                    V transition;
                    if(pass==0)transition=Ops::first(Ops::identity(),states,left[j],rank[j],j);
                    else if constexpr(Optimized)transition=Ops::second_mask(Ops::identity(),masks[j],j,n);
                    else transition=Ops::second(Ops::identity(),states,left[j],rank[j],j,n,Ops::load(a+size_t(j)*W),p[j]);
                    Ops::store(tr+size_t(j)*W,transition);
                }
            }
            #pragma omp for schedule(static)
            for(int64_t b=0;b<int64_t(S);++b){
                V current=Ops::identity();
                if(size_t(b)<blocks){
                    int lo=int(b*B),hi=int(std::min(size_t(n),(size_t(b)+1)*B));
                    for(int j=hi-1;j>=lo;--j){
                        if constexpr(Precompute)current=combine(Ops::load(tr+size_t(j)*W),current);
                        else{
                            if(pass==0)current=Ops::first(current,states,left[j],rank[j],j);
                            else if constexpr(Optimized)current=Ops::second_mask(current,masks[j],j,n);
                            else current=Ops::second(current,states,left[j],rank[j],j,n,Ops::load(a+size_t(j)*W),p[j]);
                        }
                    }
                }
                Ops::store(t+(S+b)*W,current);
            }
            size_t width=S/2;
            for(;width>=size_t(cutoff_)&&width;width/=2){
                #pragma omp for schedule(static)
                for(int64_t k=int64_t(width);k<int64_t(2*width);++k)
                    Ops::store(t+k*W,combine(Ops::load(t+2*k*W),Ops::load(t+(2*k+1)*W)));
            }
            #pragma omp single
            {
                for(size_t w=width;w;w/=2)
                    for(size_t k=w;k<2*w;++k)
                        Ops::store(t+k*W,combine(Ops::load(t+2*k*W),Ops::load(t+(2*k+1)*W)));
                Ops::store(v+W,Ops::identity());
                for(size_t w=1;w<S&&w<size_t(cutoff_);w*=2)
                    for(size_t k=w;k<2*w;++k){
                        V parent=Ops::load(v+k*W);
                        Ops::store(v+2*k*W,combine(Ops::load(t+(2*k+1)*W),parent));
                        Ops::store(v+(2*k+1)*W,parent);
                    }
            }
            for(size_t w=cutoff_;w<S;w*=2){
                #pragma omp for schedule(static)
                for(int64_t k=int64_t(w);k<int64_t(2*w);++k){
                    V parent=Ops::load(v+k*W);
                    Ops::store(v+2*k*W,combine(Ops::load(t+(2*k+1)*W),parent));
                    Ops::store(v+(2*k+1)*W,parent);
                }
            }
            #pragma omp for schedule(static)
            for(int64_t b=0;b<int64_t(blocks);++b){
                V current=Ops::load(v+(S+b)*W);alignas(64) int cells[W];
                int lo=int(b*B),hi=int(std::min(size_t(n),(size_t(b)+1)*B));
                for(int j=hi-1;j>=lo;--j){
                    Ops::store(cells,current);int code=cells[rank[j]];
                    if(pass==0){
                        p[j]=code<0?~code:n;
                        if constexpr(Optimized){
                            masks[j]=Ops::prepare_second(current,states,left[j],rank[j],p[j],n);
                            pa[j]=p[j]<n?p[j]:-1;
                        }
                        else Ops::store(a+size_t(j)*W,current);
                        if constexpr(!Precompute)current=Ops::first(current,states,left[j],rank[j],j);
                    }else{
                        nx[j]=code<0&&~code<n?~code:-1;
                        if constexpr(!Optimized)pa[j]=p[j]<n?p[j]:-1;
                        if constexpr(!Precompute){
                            if constexpr(Optimized)current=Ops::second_mask(current,masks[j],j,n);
                            else current=Ops::second(current,states,left[j],rank[j],j,n,Ops::load(a+size_t(j)*W),p[j]);
                        }
                    }
                    if constexpr(Precompute)current=combine(Ops::load(tr+size_t(j)*W),current);
                }
            }
        }
        // Parent defaults were completed before these unique-destination writes.
        bool wide_links=false;
#if defined(__AVX512F__)
        wide_links=vector_links_&&backend_!=Backend::Scalar;
#endif
        if(wide_links){
#if defined(__AVX512F__)
            const __m512i offsets=SIMD16::identity();
            #pragma omp for schedule(static)
            for(int64_t q=0;q<(int64_t(n)+15)/16;++q){
                int i=int(q*16),remaining=std::min(16,n-i);
                __mmask16 mask=__mmask16((1u<<remaining)-1);
                __m512i indexes=_mm512_maskz_loadu_epi32(mask,nx+i);
                __m512i ids=_mm512_add_epi32(offsets,_mm512_set1_epi32(i));
                mask&=_mm512_cmp_epi32_mask(indexes,_mm512_setzero_si512(),_MM_CMPINT_NLT);
                _mm512_mask_i32scatter_epi32(pa,mask,indexes,ids,4);
            }
            #pragma omp for schedule(static)
            for(int64_t q=0;q<(int64_t(n)+15)/16;++q){
                int i=int(q*16),remaining=std::min(16,n-i);
                __mmask16 valid=__mmask16((1u<<remaining)-1);
                __m512i ids=_mm512_add_epi32(offsets,_mm512_set1_epi32(i));
                __m512i parents=_mm512_maskz_loadu_epi32(valid,pa+i);
                __mmask16 roots=valid&_mm512_cmp_epi32_mask(parents,_mm512_setzero_si512(),_MM_CMPINT_LT);
                if(roots)found=_mm512_mask_reduce_max_epi32(roots,ids);
                __mmask16 is_left=_mm512_cmp_epi32_mask(ids,parents,_MM_CMPINT_LT);
                _mm512_mask_i32scatter_epi32(le,valid&~roots&is_left,parents,ids,4);
                _mm512_mask_i32scatter_epi32(ri,valid&~roots&~is_left,parents,ids,4);
                __mmask16 interior=valid&_mm512_cmp_epi32_mask(ids,_mm512_set1_epi32(n-1),_MM_CMPINT_LT);
                __m512i before=_mm512_maskz_loadu_epi32(interior,p+i);
                __m512i after=_mm512_maskz_loadu_epi32(interior,p+i+1);
                __m512i leaf_ids=_mm512_add_epi32(ids,_mm512_set1_epi32(1));
                __m512i leaves=_mm512_xor_si512(leaf_ids,_mm512_set1_epi32(-1));
                __mmask16 to_right=interior&_mm512_cmp_epi32_mask(before,after,_MM_CMPINT_LT);
                _mm512_mask_i32scatter_epi32(ri,to_right,ids,leaves,4);
                _mm512_mask_i32scatter_epi32(le,interior&~to_right,leaf_ids,leaves,4);
            }
#endif
        }else{
            #pragma omp for schedule(static)
            for(int i=0;i<n;++i)if(nx[i]>=0)pa[nx[i]]=i;
        // Operator and original-operand destinations are disjoint. Each slot
        // has one writer, so these assignments may share a parallel phase.
            #pragma omp for schedule(static)
            for(int i=0;i<n;++i){
                int parent=pa[i];
                if(parent<0)found=i;
                else if(i<parent)le[parent]=i;
                else ri[parent]=i;
                int k=i+1;
                if(k<n){if(p[i]<p[k])ri[i]=-(k+1);else le[k]=-(k+1);}
            }
        }
    }
    le[0]=-1;ri[n-1]=-(n+1);graph.root=found;
}

// One worker evaluates the same two suffix scans directly, right to left.
// Block summaries and inter-worker carries are unnecessary in this schedule.
template<class Ops> void Scan::run_single(const Input& in,Graph& graph){
    constexpr int W=Ops::W;using V=typename Ops::V;
    alignas(64) int values[W],cells[W];std::fill(values,values+W,INT_MAX);
    std::copy(in.states.begin(),in.states.end(),values);
    V states=Ops::load(values),current=Ops::identity();
    const int n=n_;actual_threads=1;
    for(int j=n-1;j>=0;--j){
        Ops::store(cells,current);int code=cells[in.rank[j]];
        pop[j]=code<0?~code:n;
        witness_masks_[j]=Ops::prepare_second(current,states,in.L[j],in.rank[j],pop[j],n);
        graph.parent[j]=pop[j]<n?pop[j]:-1;
        current=Ops::first(current,states,in.L[j],in.rank[j],j);
    }
    current=Ops::identity();
    for(int j=n-1;j>=0;--j){
        Ops::store(cells,current);int code=cells[in.rank[j]];
        next[j]=code<0&&~code<n?~code:-1;
        current=Ops::second_mask(current,witness_masks_[j],j,n);
    }
    bool wide=false;
#if defined(__AVX512F__)
    wide=vector_links()&&n>=16;
#endif
    if(wide){
#if defined(__AVX512F__)
        int found=-1;
        int *nx=next.data(),*p=pop.data(),*pa=graph.parent.data();
        int *le=graph.left.data(),*ri=graph.right.data();
            const __m512i offsets=SIMD16::identity();
            for(int64_t q=0;q<(int64_t(n)+15)/16;++q){
                int i=int(q*16),remaining=std::min(16,n-i);
                __mmask16 mask=__mmask16((1u<<remaining)-1);
                __m512i indexes=_mm512_maskz_loadu_epi32(mask,nx+i);
                __m512i ids=_mm512_add_epi32(offsets,_mm512_set1_epi32(i));
                mask&=_mm512_cmp_epi32_mask(indexes,_mm512_setzero_si512(),_MM_CMPINT_NLT);
                _mm512_mask_i32scatter_epi32(pa,mask,indexes,ids,4);
            }
            for(int64_t q=0;q<(int64_t(n)+15)/16;++q){
                int i=int(q*16),remaining=std::min(16,n-i);
                __mmask16 valid=__mmask16((1u<<remaining)-1);
                __m512i ids=_mm512_add_epi32(offsets,_mm512_set1_epi32(i));
                __m512i parents=_mm512_maskz_loadu_epi32(valid,pa+i);
                __mmask16 roots=valid&_mm512_cmp_epi32_mask(parents,_mm512_setzero_si512(),_MM_CMPINT_LT);
                if(roots)found=_mm512_mask_reduce_max_epi32(roots,ids);
                __mmask16 is_left=_mm512_cmp_epi32_mask(ids,parents,_MM_CMPINT_LT);
                _mm512_mask_i32scatter_epi32(le,valid&~roots&is_left,parents,ids,4);
                _mm512_mask_i32scatter_epi32(ri,valid&~roots&~is_left,parents,ids,4);
                __mmask16 interior=valid&_mm512_cmp_epi32_mask(ids,_mm512_set1_epi32(n-1),_MM_CMPINT_LT);
                __m512i before=_mm512_maskz_loadu_epi32(interior,p+i);
                __m512i after=_mm512_maskz_loadu_epi32(interior,p+i+1);
                __m512i leaf_ids=_mm512_add_epi32(ids,_mm512_set1_epi32(1));
                __m512i leaves=_mm512_xor_si512(leaf_ids,_mm512_set1_epi32(-1));
                __mmask16 to_right=interior&_mm512_cmp_epi32_mask(before,after,_MM_CMPINT_LT);
                _mm512_mask_i32scatter_epi32(ri,to_right,ids,leaves,4);
                _mm512_mask_i32scatter_epi32(le,interior&~to_right,leaf_ids,leaves,4);
            }
        graph.root=found;
#endif
    }else{
    for(int i=0;i<n;++i)if(next[i]>=0)graph.parent[next[i]]=i;
    for(int i=0;i<n;++i){
        int parent=graph.parent[i];
        if(parent<0)graph.root=i;
        else if(i<parent)graph.left[parent]=i;
        else graph.right[parent]=i;
        if(i+1<n){
            if(pop[i]<pop[i+1])graph.right[i]=-(i+2);
            else graph.left[i+1]=-(i+2);
        }
    }
    }
    graph.left[0]=-1;graph.right[n-1]=-(n+1);
}

void Scan::parse(const Input& in,Graph& graph,int threads){
    if(in.n()!=n_||int(in.states.size())!=c_||graph.parent.size()!=size_t(n_)||
       graph.left.size()!=size_t(n_)||graph.right.size()!=size_t(n_)||threads<1||threads>256)
        throw std::invalid_argument("workspace dimensions / thread limit");
    if(!n_){graph.root=-1;actual_threads=0;return;}
    if(optimized_witness_&&n_==1){
        // Both exclusive suffixes are empty identities.
        pop[0]=1;next[0]=-1;graph.parent[0]=-1;
        graph.left[0]=-1;graph.right[0]=-2;graph.root=0;
        actual_threads=1;return;
    }
    if(optimized_witness_)threads=std::min(threads,int(blocks_));
    auto dispatch=[&](auto operations){
        using Ops=decltype(operations);
        if(optimized_witness_){
            if(threads==1&&!precompute_)run_single<Ops>(in,graph);
            else if(precompute_)run<Ops,true,true>(in,graph,threads);
            else run<Ops,false,true>(in,graph,threads);
        }else{
            if(precompute_)run<Ops,true,false>(in,graph,threads);
            else run<Ops,false,false>(in,graph,threads);
        }
    };
    switch(backend_){
#if defined(__AVX__)
    case Backend::SIMD4:dispatch(SIMD4{});break;
#endif
#if defined(__AVX2__)
    case Backend::SIMD8:dispatch(SIMD8{});break;
#endif
#if defined(__AVX512F__)
    case Backend::SIMD16:dispatch(SIMD16{});break;
#endif
    default:
        if(width_==4)dispatch(Scalar<4>{});
        else if(width_==8)dispatch(Scalar<8>{});
        else dispatch(Scalar<16>{});
    }
}

Stack::Stack(int n,bool diagnostics):stack_(n){
    if(diagnostics){pop.resize(n);next.resize(n);last_.resize(size_t(n)+1);}
}
size_t Stack::scratch_bytes()const{return stack_.size()*sizeof(uint64_t)+(pop.size()+next.size()+last_.size())*sizeof(int);}
template<bool Diagnostics> void Stack::run(const Input& in,Graph& g){
    int n=in.n(),top=0;auto* st=stack_.data();
    int* pa=g.parent.data();int* le=g.left.data();int* ri=g.right.data();
    const int* L=in.L.data();const int* R=in.R.data();
    for(int j=0;j<n;++j){
        int value=-(j+1),bound=L[j];
        while(top&&int32_t(st[top-1]>>32)>bound){
            int i=int(uint32_t(st[--top]));
            ri[i]=value;if(value>=0)pa[value]=i;value=i;
            if constexpr(Diagnostics)pop[i]=j;
        }
        le[j]=value;if(value>=0)pa[value]=j;
        st[top++]=(uint64_t(uint32_t(R[j]))<<32)|uint32_t(j);
    }
    int value=-(n+1);
    while(top){
        int i=int(uint32_t(st[--top]));ri[i]=value;if(value>=0)pa[value]=i;value=i;
        if constexpr(Diagnostics)pop[i]=n;
    }
    g.root=value;if(n)pa[value]=-1;
    if constexpr(Diagnostics){
        std::fill(last_.data(),last_.data()+n+1,-1);
        for(int i=n-1;i>=0;--i){next[i]=last_[pop[i]];last_[pop[i]]=i;}
    }
}
void Stack::parse(const Input& in,Graph& g,bool diagnostics){
    if(stack_.size()!=size_t(in.n())||g.parent.size()!=size_t(in.n()))throw std::invalid_argument("stack workspace size");
    if(!in.n()){g.root=-1;return;}
    if(diagnostics){if(pop.size()!=size_t(in.n()))throw std::invalid_argument("diagnostic workspace missing");run<true>(in,g);}
    else run<false>(in,g);
}
bool equal_graph(const Graph& a,const Graph& b){
    if(a.root!=b.root||a.parent.size()!=b.parent.size())return false;
    size_t bytes=a.parent.size()*sizeof(int);
    return !bytes||(!std::memcmp(a.parent.data(),b.parent.data(),bytes)&&
        !std::memcmp(a.left.data(),b.left.data(),bytes)&&!std::memcmp(a.right.data(),b.right.data(),bytes));
}
uint64_t checksum(const Graph& g){
    uint64_t h=uint32_t(g.root);
    for(size_t i=0;i<g.parent.size();++i)
        h=(h^uint32_t(g.parent[i]))*0x100000001b3ULL+uint32_t(g.left[i])+3*uint64_t(uint32_t(g.right[i]));
    return h;
}
} // namespace peak

extern "C" int peak_simd_supported() noexcept {
    int mask=0;
    for(int k=0;k<=4;++k)if(peak::available(static_cast<peak::Backend>(k)))mask|=1<<k;
    return mask;
}

extern "C" int peak_simd_graph(int n,int c,const int* states,const int* left,const int* right,
    int threads,int backend,int block,int* pop,int* next,int* parent,int* lchild,int* rchild,int* root) noexcept {
    try{
        if(n<0||n>INT_MAX-2||c<1||c>16||backend<0||backend>4)return 1;
        peak::Input in;in.states.assign(states,states+c);in.L.assign(left,left+n);in.R.assign(right,right+n);in.rank.resize(n);
        for(int i=0;i<n;++i)in.rank[i]=int(std::lower_bound(in.states.begin(),in.states.end(),right[i])-in.states.begin());
        in.validate();peak::Graph g;g.resize(n);
        peak::Scan scan(in,static_cast<peak::Backend>(backend),block);scan.parse(in,g,threads);
        if(n){size_t bytes=size_t(n)*sizeof(int);
            std::memcpy(pop,scan.pop.data(),bytes);std::memcpy(next,scan.next.data(),bytes);
            std::memcpy(parent,g.parent.data(),bytes);std::memcpy(lchild,g.left.data(),bytes);std::memcpy(rchild,g.right.data(),bytes);}
        *root=g.root;return 0;
    }catch(...){return 2;}
}
