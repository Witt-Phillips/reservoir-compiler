// src/main.rs
mod reservoir;

use crate::reservoir::Reservoir;

fn main() {    
    let mut ex_res = Reservoir::new(5, 3, 4);
    ex_res.print();

    while !ex_res.converged {
        ex_res.run();
    }
    
    println!("converged!");
    ex_res.readout();
}
