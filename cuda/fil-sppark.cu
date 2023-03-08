#include <cuda.h>

// Imports from sppark. The order matters.
#include <ff/bls12-381.hpp>
#include <ntt/ntt.cuh>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
// Those definitions are needed by `msm/pippenger.cuh`.
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;
#include <msm/pippenger.cuh>

#ifndef __CUDA_ARCH__
extern "C" RustError mult_pippenger(jacobian_t<fp_t>* out,
                                    const affine_t points[],
                                    size_t npoints,
                                    const scalar_t scalars[]) {
   return mult_pippenger<bucket_t>(out, points, npoints, scalars, false);
}

//extern "C" RustError compute_ntt(fr_t* inout, uint32_t lg_domain_size,
//                     NTT::InputOutputOrder ntt_order,
//                     NTT::Direction ntt_direction,
//                     NTT::Type ntt_type)
extern "C" RustError compute_ntt(fr_t* inout, uint32_t lg_domain_size) {
   // As for MSM, use the first available device for now.
   auto& gpu = select_gpu(0);
   return NTT::Base(gpu, inout, lg_domain_size, NTT::InputOutputOrder::NN,
                    NTT::Direction::forward, NTT::Type::standard);
}
#endif
