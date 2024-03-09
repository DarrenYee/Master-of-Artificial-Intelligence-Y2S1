% power generation
int: T;                 % decades
array[1..T] of int: e;  % expected requirements
array[1..T] of int: a;  % current production

array[1..T] of var 0..infinity: N; % number of nuclear power plants built each decade
array[1..T] of var 0..infinity: C; % number of coal power plants built each decade
array[1..T] of var 0..infinity: S; % number of solar power plants built each decade

array[1..T] of var 0..infinity: Nenergy; % nuclear power available in each decade
array[1..T] of var 0..infinity: Cenergy; % coal power available in each decade
array[1..T] of var 0..infinity: Senergy; % solar power available in each decade

constraint forall (i in 1..T) (Nenergy[i] = sum ( j in max(1, i - 5)..i)(N[j] * 4));

constraint forall (i in 1..T) (Cenergy[i] = sum ( j in max(1, i - 1)..i)(C[j] * 1));

constraint forall (i in 1..T) (Senergy[i] = sum ( j in max(1, i - 2)..i)(S[j] * 1));

constraint forall (i in 1..T) (a[i] + Nenergy[i] + Cenergy[i] + Senergy[i] >= e[i]);

constraint forall (i in 1..T) ((a[i] + Nenergy[i] + Cenergy[i] + Senergy[i])*0.4 >= Nenergy[i]);

constraint forall (i in 1..T) ((a[i] + Nenergy[i] + Cenergy[i] + Senergy[i])*0.2 <= Senergy[i]);

constraint sum (Cenergy) <= 10;
var 0..infinity: cost;  % costs of building all new power plants
constraint cost = sum (N)*10 + sum (C) + sum (S);
solve minimize cost;

 
                           
