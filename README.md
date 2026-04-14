# Learning Racket-Ball Bounce Dynamics Across Diverse Rubbers for Robotic Table Tennis

<p align="center">
  <a href="https://arxiv.org/abs/2604.11349">
    <img src="https://img.shields.io/badge/arXiv-2604.11349-b31b1b.svg"/>
  </a>
  <a href="https://arxiv.org/pdf/2604.11349.pdf">
    <img src="https://img.shields.io/badge/PDF-Download-blue.svg"/>
  </a>
</p>


Modeling racket-ball interactions in robotic table tennis across a wide range of rubber types.  
This work combines a physics-based contact model with Gaussian Processes to capture nonlinear, state-dependent bounce dynamics while preserving physical interpretability.


<p align="center">
  <img src="rackets.jpeg" alt="Racket configurations" width="500"/>
</p>

---

## Contributions

- Dataset of 8,194 racket-ball bounce events across 10 racket configurations  
- Analysis of rubber-dependent bounce dynamics and their dependence on impact conditions  
- Gaussian Process framework for learning state-dependent physical parameters with uncertainty estimates  
- Improved prediction accuracy of post-impact velocity and spin compared to standard baselines  
- Online adaptation of racket dynamics with few observations  

---

## Status 🚧

This repository is under construction. Code and dataset will be released soon.


## Citation

```bibtex
@misc{gossard2026learningracketballbouncedynamics,
      title={Learning Racket-Ball Bounce Dynamics Across Diverse Rubbers for Robotic Table Tennis}, 
      author={Thomas Gossard},
      year={2026},
      eprint={2604.11349},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2604.11349}, 
}
