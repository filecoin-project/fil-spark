use std::mem;

use blst::{blst_p1, blst_p1_affine, blst_scalar};
use blstrs::{G1Affine, G1Projective, Scalar};

sppark::cuda_error!();

extern "C" {
    fn mult_pippenger(
        out: *mut blst_p1,
        points: *const blst_p1_affine,
        npoints: usize,
        scalars: *const blst_scalar,
    ) -> cuda::Error;

    fn compute_ntt(
        //inout: *mut core::ffi::c_void,
        inout: *mut blst_scalar,
        lg_domain_size: u32,
    ) -> cuda::Error;
}

/// A multi-scalar multiplication (MSM).
///
/// Also known as multi-exponentiation (multiexp).
pub fn multi_scalar_multiplication(points: &[G1Affine], scalars: &[[u8; 32]]) -> G1Projective {
    assert_eq!(points.len(), scalars.len());

    let mut ret = blst_p1::default();
    let err = unsafe {
        mult_pippenger(
            &mut ret as *mut _ as *mut blst_p1,
            &points[0] as *const _ as *const blst_p1_affine,
            points.len(),
            &scalars[0] as *const _ as *const blst_scalar,
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
    // TODO vmx 2023-03-07: add such a converson to blstrs directly (in a safe way).
    unsafe { mem::transmute::<blst_p1, G1Projective>(ret) }
}

/// A number theoretic transform (NTT).
///
/// It's the integer version of a fast Fourier transform (FFT).
//pub fn number_theoretic_transform(input: &mut [[u8; 32]], lg_domain_size: u32) {
pub fn number_theoretic_transform(input: &mut [Scalar], lg_domain_size: u32) {
    unsafe { compute_ntt(input as *mut _ as *mut blst_scalar, lg_domain_size) };
}
