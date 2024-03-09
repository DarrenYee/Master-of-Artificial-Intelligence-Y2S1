% power generation
int: T;                 % decades

array[1..T] of 0..infinity: N; % number of nuclear power plants built each decade
array[1..T] of 0..infinity: C; % number of coal power plants built each decade
array[1..T] of 0..infinity: S; % number of solar power plants built each decade

array[1..T] of var 0..infinity: Nenergy; % nuclear power available in each decade
array[1..T] of var 0..infinity: Cenergy; % coal power available in each decade
array[1..T] of var 0..infinity: Senergy; % solar power available in each decade

constraint forall (i in 1..T) (Nenergy[i] = sum ( j in max(1, i - 5)..i)(N[j] * 4));

constraint forall (i in 1..T) (Cenergy[i] = sum ( j in max(1, i - 1)..i)(C[j] * 1));

constraint forall (i in 1..T) (Senergy[i] = sum ( j in max(1, i - 2)..i)(S[j] * 1));

