
enum PLAYER = {Ant, Bee, Chu, Deb, Eve, Fin, Ged, Hel, Ila, Jan, Kim};

enum CAPTAIN = {Xavier, Yuri, Zena};


array[CAPTAIN,PLAYER] of int: value = 
   [| 2, 5, 6, 8, 9, 5, 8, 7, 7, 4, 6
    | 9, 8, 4, 7, 6, 4, 5, 3, 5, 5, 7
    | 8, 4, 3, 3, 6, 2, 5, 5, 3, 2, 5 |];

var set of PLAYER: xavier;
var set of PLAYER: yuri;
var set of PLAYER: zena;

var set of PLAYER: goalies = {Ant, Bee};
var set of PLAYER: defense = {Chu, Deb, Eve, Fin};
var set of PLAYER: offense = {Ged, Hel, Ila, Jan, Kim};

constraint (card(xavier intersect yuri) <= 2);
constraint (card(xavier intersect zena) <= 2);

constraint (card (xavier intersect goalies) >= 1);
constraint (card (zena intersect goalies) >= 1);
constraint (card (yuri intersect goalies) >= 1);

constraint (card (xavier intersect defense) >= 2);
constraint (card (zena intersect defense) >= 2);
constraint (card (yuri intersect defense) >= 2);

constraint (card (xavier intersect offense) >= 2);
constraint (card (zena intersect offense) >= 2);
constraint (card (yuri intersect offense) >= 2);

constraint card(xavier) = 6;
constraint card(yuri) = 6;
constraint card(zena) = 6;

var int: perceived_value_X;
constraint perceived_value_X = sum (p in xavier) (value[Xavier,p]);

var int: perceived_value_Y;
constraint perceived_value_Y = sum (p in yuri) (value[Yuri,p]);

var int: perceived_value_Z;
constraint perceived_value_Z = sum (p in zena) (value[Zena,p]);

solve maximize perceived_value_X + perceived_value_Y + perceived_value_Z;

output ["xavier = \(xavier);\nyuri = \(yuri);\nzena = \(zena);\n"];

