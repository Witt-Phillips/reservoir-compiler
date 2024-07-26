// src/reservoir.rs

extern crate nalgebra as na;
use na::{DMatrix, DVector};

//const GAMMA: f64 = 0.05;
const CONVERGENCE_CONSTANT: f64 = 1e-5;

pub struct Reservoir {
    r: DVector<f64>,
    x: DVector<f64>,
    d: DVector<f64>,
    a: DMatrix<f64>,
    b: DMatrix<f64>,
    w: DMatrix<f64>,
    pub converged: bool,
}

/* 
k: input dim
n: latent dim
m: output dim
*/
impl Reservoir {
    /* Zeroed constructor. */
    pub fn new(k: usize, m: usize, n: usize) -> Self {
        Reservoir {
            r: DVector::zeros(n),
            x: DVector::zeros(k),
            d: DVector::zeros(n),
            a: DMatrix::zeros(n, n),
            b: DMatrix::zeros(n, k),
            w: DMatrix::zeros(m, n),
            converged: false,
        }
    }
    
    /* r_t+1 = tanh(Ar + Bx + d) */
    pub fn run(&mut self) {
        let prev_r = self.r.clone();
        self.r = (self.a.clone() * self.r.clone() + self.b.clone() * self.x.clone() + self.d.clone()).map(|v| v.tanh());

        if (prev_r - self.r.clone()).abs().max() < CONVERGENCE_CONSTANT {
            self.converged = true;
        }
    }

    /* O = Wr */
    pub fn readout(&self) -> DVector<f64> {
        &self.w * &self.r
    }

    pub fn print(&self) {
        println!("Printing Reservoir:");
        println!("r: {:?}", self.r);
        println!("x: {:?}", self.x);
        println!("d: {:?}", self.d);
        println!("A: {:?}", self.a);
        println!("B: {:?}", self.b);
        println!("W: {:?}", self.w);
        println!("converged: {:?}", self.converged);
    }
}

impl Clone for Reservoir {
    fn clone(&self) -> Self {
        Reservoir {
            r: self.r.clone(),
            x: self.x.clone(),
            d: self.d.clone(),
            a: self.a.clone(),
            b: self.b.clone(),
            w: self.w.clone(),
            converged: self.converged,
        }
    }
}
