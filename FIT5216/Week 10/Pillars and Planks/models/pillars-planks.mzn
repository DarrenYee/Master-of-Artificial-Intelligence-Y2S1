int: planks;
set of int: PLANK = 1..planks;
array[PLANK] of int: plank_width;
int: pillars;
set of int: PILLAR = 1..pillars;
array[PILLAR] of int: pillar_height;
array[PILLAR] of int: pillar_width;

int: available_width;
int: available_height;
set of int: WIDTH = 0..available_width-1;
set of int: HEIGHT = 0..available_height-1; 

array[PLANK] of var WIDTH: xk;
array[PLANK] of var HEIGHT: yk;
array[PILLAR] of var WIDTH: xr;
array[PILLAR] of var HEIGHT: yr;
var 0..available_height: height;
   
include "diffn.mzn";
include "cumulative.mzn";

constraint diffn (xk ++xr,yk++yr,plank_width ++ pillar_width,[1|p in PLANK] ++ pillar_height );
constraint forall (p in PLANK) (xk[p] + plank_width[p] <= available_width);
constraint forall (p in PILLAR) (xr[p] + pillar_width[p] <= available_width);
constraint forall (p in PLANK) (yk[p] + 1 <= available_height);
constraint forall (p in PILLAR) (yr[p] + pillar_height[p] <= available_height);

array[PLANK] of var PILLAR: left;
array[PLANK] of var PILLAR: right;
constraint forall (p in PLANK) (xr[left[p]] <= xk[p] /\ xr[left[p]] + pillar_width[left[p]] > xk[p] /\ yk[p] = yr[left[p]] + pillar_height[left[p]]);
constraint forall (p in PLANK) (xr[right[p]] <= xk[p] + plank_width[p] -1 /\ xr[right[p]] + pillar_width[right[p]] > xk[p] + plank_width[p] -1 /\ yk[p] = yr[right[p]] + pillar_height[right[p]]);

set of int :PLANK0 = 0..planks; % 0 because of the dummy plnak below
array[PLANK0] of int: plank_widthx = array1d(PLANK0,[available_width] ++ plank_width); %add the available width as the width of dummy 
array[PLANK0] of var WIDTH: xkx = array1d(PLANK0,[0] ++ xk); %width of dummy plank = 0
array[PLANK0] of var -1..available_height-1: yky = array1d(PLANK0,[-1] ++ yk); %%dummy plank starts at -1

array[PILLAR] of var PLANK0: support;
constraint forall (p in PILLAR) (xr[p] >= xkx[support[p]] /\ xr[p] + pillar_width[p] <= xkx[support[p]] + plank_widthx[support[p]] /\ yr[p]= yky[support[p]] + 1 );

constraint height = max([yk[p] +1 | p in PLANK] ++ [yr[p] + pillar_height[p] | p in PILLAR]);
%%%% output for grader   
output ["height = \(height);\n"];          
output ["xk = \(xk);\nyk = \(yk);\nxr = \(xr);\nyr = \(yr);\n"];             
                                          

                              
                                                                        
                  
                  