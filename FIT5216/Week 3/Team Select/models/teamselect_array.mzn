
enum PLAYER = {Ant, Bee, Chu, Deb, Eve, Fin, Ged, Hel, Ila, Jan, Kim};

enum CAPTAIN = {Xavier, Yuri, Zena};

array[CAPTAIN,PLAYER] of int: value = 
   [| 2, 5, 6, 8, 9, 5, 8, 7, 7, 4, 6
    | 9, 8, 4, 7, 6, 4, 5, 3, 5, 5, 7
    | 8, 4, 3, 3, 6, 2, 5, 5, 3, 2, 5 |];

set of int: POS = 1..6;

array[CAPTAIN,POS] of var PLAYER: team;

constraint forall (c in CAPTAIN) (sum(i in POS) (team[c,i]) < 10);

output ["team = array2d(CAPTAIN,POS,\(team));\n"];

