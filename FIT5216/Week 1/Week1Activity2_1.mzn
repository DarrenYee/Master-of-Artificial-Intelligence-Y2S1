var 1..8: C1;  
var 1..8: S1;  
var 1..8: C2;
var 1..8: S2;                 
var 1..8: S3;
var 1..8: C3;  
var 1..8: S4;  
var 1..8: C4;

constraint C1 + C2 = S1;
constraint C1 + C3 = S2;
constraint C2 + C4 = S3;
constraint C3 + C4 = S4;

include "alldifferent.mzn";   
constraint alldifferent ([C1,S1,C2,S2,S3,C3,S4,C4]);     



