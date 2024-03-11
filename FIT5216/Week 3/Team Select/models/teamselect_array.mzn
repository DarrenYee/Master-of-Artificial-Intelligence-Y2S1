
enum PLAYER = {Ant, Bee, Chu, Deb, Eve, Fin, Ged, Hel, Ila, Jan, Kim};

enum CAPTAIN = {Xavier, Yuri, Zena};

array[CAPTAIN,PLAYER] of int: value = 
   [| 2, 5, 6, 8, 9, 5, 8, 7, 7, 4, 6
    | 9, 8, 4, 7, 6, 4, 5, 3, 5, 5, 7
    | 8, 4, 3, 3, 6, 2, 5, 5, 3, 2, 5 |];

set of int: POS = 1..6;

var set of PLAYER: goalies = {Ant, Bee};
var set of PLAYER: defense = {Chu, Deb, Eve, Fin};
var set of PLAYER: offense = {Ged, Hel, Ila, Jan, Kim};

array[CAPTAIN,POS] of var PLAYER: team;

constraint  (sum (p1 in POS,p2 in POS) (team[Xavier,p1] = team[Yuri,p2]) <= 2);
constraint  (sum (p1 in POS,p2 in POS) (team[Xavier,p1] = team[Zena,p2]) <= 2);

constraint forall (c in CAPTAIN) (sum (p in POS) (team[c,p] in goalies) >= 1);
constraint forall (c in CAPTAIN) (sum (p in POS) (team[c,p] in defense) >= 2);
constraint forall (c in CAPTAIN) (sum (p in POS) (team[c,p] in offense) >= 2);

constraint forall (c in CAPTAIN, play in PLAYER) (sum (p in POS) (team[c,p] = play) <= 1);

var int: perceived_value_X;
constraint perceived_value_X = sum (p in POS) (value[Xavier,team[Xavier,p]]);

var int: perceived_value_Y;
constraint perceived_value_Y = sum (p in POS) (value[Yuri,team[Yuri,p]]);

var int: perceived_value_Z;
constraint perceived_value_Z = sum (p in POS) (value[Zena,team[Zena,p]]);

solve maximize perceived_value_X + perceived_value_Y + perceived_value_Z;

% constraint forall (c in CAPTAIN, p in POS) (sum (d in defense) (team[c,p] = d) >= 2);
% constraint forall (c in CAPTAIN, p in POS) (sum (o in offense) (team[c,p] = o) >= 1);

output ["team = array2d(CAPTAIN,POS,\(team));\n"];

