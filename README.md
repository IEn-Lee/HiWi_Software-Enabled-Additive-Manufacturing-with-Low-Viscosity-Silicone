# RTV-2 Silicone G-code Optimizer

**Model-Based Process Optimization for Low-Viscosity Silicone Additive Manufacturing**

A Python tool for converting FDM-style G-code into toolpaths for RTV-2 silicone extrusion printing, incorporating residence-time modeling and curing-aware process optimization.

**Student Research Assistant — I-En Lee**  
Institute for Factory Automation and Production Systems (FAPS)  
Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)

## Project Documentation

- **Illustrated Overview:** [Explore the project with figures and explanations](https://github.com/IEn-Lee/Research-Assistant_Software-Enabled-Additive-Manufacturing-with-Low-Viscosity-Silicone/blob/main/Silicone%20Additive%20Manufacturing%20Portfolio.pdf)

*If GitHub fails to display the PDF preview, please download the file and open it locally.*


---


## Overview

This research focused on improving **extrusion-based additive manufacturing with low-viscosity silicone** through process modeling, software-based optimization, and experimental validation.

My work connected material residence time and thermal history to changes in material behavior, using these relationships to improve material conditioning consistency and printing performance.


## Process Challenge

The material state of RTV-2 silicone evolves after mixing and depends on its residence time and thermal history. During printing, material associated with different extrusion segments may experience different residence times, resulting in inconsistent material conditions at deposition.

The objective of this work was to reduce these differences through model-based process optimization, helping maintain more consistent material conditions and printed-segment quality throughout the toolpath.


## My Contributions

- Developed a Python tool to process FDM-style G-code for RTV-2 silicone extrusion printing.
- Analyzed toolpaths and tracked material residence time across extrusion segments.
- Modeled material-state evolution based on residence time and thermal history.
- Applied PID-based optimization to adjust segment-level feedrates and improve residence-time consistency.
- Conducted physical printing tests and evaluated the effects of optimization on the printed samples.

## 1. Improving Consistency During Extrusion

A key objective was to maintain more consistent material conditions throughout the printing process. Variations in residence time and thermal history can affect the state of the silicone as it reaches the extrusion outlet, influencing flow behavior and print quality.

I developed a predictive process-modeling framework and applied PID-based optimization to improve consistency across extrusion segments.

<p align="center">
  <img src="Images/Silicone_Material_Conditioning_Stability.png" alt="Residence-time stability index before and after process optimization" width="850">
</p>

*Figure 1. Comparison of the residence-time stability index across extrusion segments before and after optimization.*

### How to Read the Curve

The Residence-Time Stability Index represents the consistency of material residence time across extrusion segments. A flatter profile indicates that material associated with different segments has experienced more similar residence times.

**The ideal profile is a horizontal line.** The objective is to minimize variation between segments around a suitable residence-time level, rather than to maximize the index value.

- **Before optimization (blue):** The profile drops sharply and varies across the printing sequence, indicating differences in residence time between segments.
- **After optimization (red):** The profile remains approximately horizontal, indicating more consistent residence times across segments.

More consistent residence times help maintain a more uniform material state during deposition, supporting more consistent printed-segment quality. The physical printing results below provide a complementary visual assessment of the optimization.

## 2. Evaluating the Printed Results

To evaluate how the process optimization translated into practical printing outcomes, I conducted printing tests and compared the resulting silicone samples.

<p align="center">
  <img src="Images/Silicone_Printing_Before_After.png" alt="Silicone printing results before optimization on the left and after optimization on the right" width="800">
</p>

*Figure 2. Silicone samples before optimization (left) and after optimization (right).*

The sample produced **before optimization** shows a more irregular outline and uneven spreading. The sample produced **after optimization** has a more regular shape and more clearly defined edges.

Together, the two figures connect process behavior with the physical printing outcome: Figure 1 shows improved residence-time consistency, while Figure 2 shows a more regular printed outline after optimization.


## Outcome

The work combined **process modeling, optimization software, and physical printing experiments** to improve silicone extrusion printing.

The main outcomes were:

- Improved material conditioning consistency.
- Reduced extrusion instability during printing.
- Improved shape regularity in the illustrated printed sample.
- Experimental validation of the model-based optimization software.
- A foundation for further process optimization in silicone additive manufacturing.

**Core skills:** Python · G-code processing · Toolpath analysis · Residence-time modeling · Curing-aware process optimization · PID-based optimization · Silicone additive manufacturing · Experimental validation


--


## Material and Process Context

This project investigates extrusion-based additive manufacturing using low-viscosity, two-component RTV-2 silicone. Its focus is on how material residence time and thermal history influence the material state during printing, and how these effects can be considered when generating and optimizing toolpaths.

The optimization approach links software-defined printing parameters to material behavior and evaluates the resulting changes through physical printing experiments.
