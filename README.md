# Lab Value Prediction

This project contains the PoC for the full model available at [https://github.com/akshat-git/StanfordLabPredictionVLM](url)

  Clinical laboratory values derived from routine blood tests are critical indicators of patient health and inform care decisions. While many clinical AI models target long-term outcomes, such as mortality or readmission, these endpoints are often confounded by the dynamic effects of medical intervention, external environmental factors, and the delayed nature of these events. Such limitations make mortality prediction less actionable for clinicians making high-stakes, time-sensitive treatment decisions. In contrast, lab values, often taken daily, provide early, interpretable, and measurable physiological signals that reflect real-time patient status. Importantly, lab value trajectories can inform counterfactual simulations of treatment response, enabling more robust and precise mortality modeling.​
 
  Preliminary results on a representative test set of n = 4000 examples demonstrate that the model predicts continuous lab values with under 5% absolute error in 93.4% of cases, with a Mean Absolute Percentage Error (MAPE) of 2.39%, including key markers such as creatinine, hemoglobin, and white blood cell count. These findings validate our hypothesis and demonstrate that intermediate physiological states can be inferred from clinical context using multimodal learning. This framework will enable interpretable, real-time assessment of patient health and decision support in dynamic treatment settings.

**Research Poster:**
[https://docs.google.com/presentation/d/1aXA8aRtxsOSOkjYI64NXlrym4rc4nJtL/edit?usp=sharing&ouid=115539732780824649086&rtpof=true&sd=true](url)

**Acknowledgements:**
This research was made possible thanks to the Stanford Institutes of Medicine Research (SIMR) Program, funding from the Strober Family Scholars Fund, and the generous support from members of the Daneshjou Lab at Stanford University. 
