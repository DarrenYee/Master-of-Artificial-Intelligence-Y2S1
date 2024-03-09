var 1..8: C1;  
var 1..8: S1;  
var 1..8: C2;
var 1..8: S2;                 
var 1..8: S3;
var 1..8: C3;  
var 1..8: S4;  
var 1..8: C4;

constraint C1 = 1;
constraint C3 = 5;

constraint abs(C1 - S1) != 1 /\ abs(S1 - C2) != 1 /\ abs(C2 - S2) != 1 /\
            abs(S2 - C3) != 1 /\ abs(C3 - S3) != 1 /\ abs(S3 - C4) != 1 /\
            abs(C4 - S4) != 1 /\ abs(S4 - C1) != 1;
            
include "alldifferent.mzn";   
constraint alldifferent ([C1,S1,C2,S2,S3,C3,S4,C4]);             