# level_set_demo

* use a standalone tool to create a spline, along which we will create "epi" cells to represent a membrane
* compute a level set "substrate" field (to be used later perhaps)
* create 2 cells that simply chemotax to the top of the domain and, as they contact epi cells, form a spring attachment and drag them along
* the "contact with" signal and "spring attachment" behavior is defined with a single rule

## First attempt

<img src=.\images\spline_tool.png width="30%">

<img src=.\images\bm_dist.png width="50%">

<img src=.\images\punch_thru_2cells.png width="50%">

<img src=.\images\rule1_studio.png width="70%">
<img src=.\images\rule1_plot.png width="40%">

## Second attempt
* renamed "epi" to "basal" cell type
* added "luminal" - approximately aligning (offset) with "basal"

<img src=.\images\basal_luminal_v0_t0.png width="40%"><img src=.\images\basal_luminal_v0.png width="40%">

