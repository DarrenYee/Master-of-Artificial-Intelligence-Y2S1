var 1..8 : a;
var 1..8 : b;
var 1..8 : c;
var 1..8 : d;
var 1..8 : e;
var 1..8 : f;
var 1..8 : g;
var 1..8 : h;


constraint a != b + 1 /\ a != b - 1 /\ a != c + 1 /\ a != c -1 /\ a != d + 1 /\ a != d-1;

constraint b != a - 1 /\ b != a + 1 /\ b != c + 1 /\ b != c - 1 /\ b != e + 1 /\ b != e - 1;

constraint c != b - 1 /\ c != b + 1 /\ c != a -1 /\ c != a + 1 /\ c != d - 1 /\ c != d + 1 /\ c != f - 1 /\ c != f + 1 /\ c != e - 1 /\ c != e + 1 /\ c != g - 1 /\ c != g + 1;

constraint d != a - 1 /\ c != a + 1 /\ d != c -1 /\ d != c + 1 /\ d != g - 1 /\ d != g + 1;

constraint e != b - 1 /\ e != b + 1 /\ e != f -1 /\ e != f + 1 /\ e != h - 1 /\ e != h + 1;

constraint f != e - 1 /\ f != e + 1 /\ f != b -1 /\ f != b + 1 /\ f != c - 1 /\ f != c + 1 /\ f != d - 1 /\ f != d + 1 /\ f != g - 1 /\ f != g + 1 /\ f != h - 1 /\ f != h + 1;

constraint h != e - 1 /\ h != e + 1 /\ h != f -1 /\ h != f + 1 /\ h != g - 1 /\ d != g + 1;

include "alldifferent.mzn";
constraint alldifferent([a,b,c,d,e,f,g,h]);




