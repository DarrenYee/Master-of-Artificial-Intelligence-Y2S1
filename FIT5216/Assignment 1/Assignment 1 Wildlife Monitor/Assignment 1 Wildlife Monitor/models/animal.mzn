enum LOC; % set of locations
LOC: base; % base location where information is collected
constraint assert(base = min(LOC),"base is not the first mentioned location\n");
int: n; % number of traps available
array[LOC] of int: wild; % amount of interesting wildlife at each location
array[LOC] of int: cost; % cost to setup trap at location, -ve means unavailable
constraint assert(cost[base] = 0, "Cost of base location is not 0\n");
array[LOC,LOC] of int: d; % distance from one location to another (symmetric)
constraint forall(l1, l2 in LOC where l1 <= l2)(assert(d[l1,l2] = d[l2,l1], "distance is not symmetric at \(l1) \(l2)\n"));
constraint forall(l in LOC)(assert(d[l,l] = 0, "self distance for \(l) is not zero\n"));
int: move; % movement distance of animals
int: link; % wireless link maximum communication distance;
int: mind; % mininum distance between two traps
int: opcost; % cost to maintain/operate camera trap
int: budget; % total budget for traps
var int: total_wild;
var int: actual_traps;

set of int: PLACE0 = 0..n;
set of int: PLACE = 1..n;
array[PLACE0] of var LOC: x; % location where camera trap is placed (except 0)
array[PLACE] of var PLACE0: s; % location to send info (part C only)
var set of LOC: covered; % set of locations which represents t  he covered
var set of LOC: uncovered;

constraint actual_traps <= n  /\ actual_traps >= 1; % actual number of traps, works similarly to the Week 3 workshop Size value


% PART 1 / PART 2
constraint x[0] = base; % first location has to be base
constraint forall(i in 1..actual_traps) (cost[x[i]] >= 0); % every other location cannot be base and that cost for the chosen trap cannot be negative
constraint forall (i in 1..actual_traps,j in 1..actual_traps where i!=j) (d[x[i],x[j]] >= mind); % double for loop through the matrix, ignoring itself and making sure that each locations within x are at least min distance away
constraint forall (i in (actual_traps + 1)..n) (x[i] = base /\ s[i] = 0);
constraint forall (i in 1..actual_traps) (x[i] != base);
constraint forall (j in covered) (exists (i in 1..actual_traps)(d[x[i],j] <= move)); 
constraint uncovered = LOC diff covered;
constraint forall (j in uncovered, i in 1..actual_traps) (d[x[i],j] > move);

% constraint forall(i in 1..actual_traps - 1)(x[i] < x[i+1]); 

% PART 2 / PART 3
constraint forall (i in 1..actual_traps where d[x[i],base] <= link) (s[i] = 0);

constraint forall (i in 1..actual_traps where d[x[i],base] > link)((d[x[i],x[s[i]]] <= link /\ s[i] != i));

var int: my_budget;
constraint my_budget = sum (i in 1..actual_traps) (opcost + cost[x[i]]);
constraint my_budget <= budget;
 
constraint total_wild = sum (j in covered) (wild[j]);

constraint forall(i in 1..actual_traps - 1)(s[i] < i);

solve maximize total_wild;

