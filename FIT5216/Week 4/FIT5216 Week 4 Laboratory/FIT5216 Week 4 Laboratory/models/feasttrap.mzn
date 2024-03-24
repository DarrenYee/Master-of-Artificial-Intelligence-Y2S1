% Beauty Trap
% Create the most magnificient feast 
enum DISH;
enum TASTE = {spicy, sour, salty, sweet, umami, bland};
enum TEMP = {hot, cold, warm};

array[DISH] of TASTE: taste;
array[DISH] of TEMP: temp;
array[DISH] of bool: heavy;
array[DISH] of int: value;

int: len; % length of banquet
set of int: COURSE = 1..len;

array[COURSE] of var DISH: dish;

var 0..infinity: obj;

include "alldifferent.mzn";
constraint alldifferent(dish);
array[1..len-1] of var int: temp1;

% No two dishes of the same taste in a row

constraint forall (i in 1..len-1) ((taste[dish[i]] != taste[dish[i+1]]) /\ ((taste[dish[i]] = spicy) -> (taste[dish[i+1]] = bland \/ taste[dish[i+1]] = sweet)) /\ ((taste[dish[i]] = sour) -> (taste[dish[i+1]] = bland \/ taste[dish[i+1]] = umami)) /\ ((taste[dish[i]] = sweet) -> (taste[dish[i+1]] != spicy /\ taste[dish[i+1]] != umami)) /\ (heavy[dish[i]] -> heavy[dish[i+1]] = false));

% First dish = salty, Last dish = sweet

constraint (taste[dish[1]] = salty) /\ (taste[dish[len]] = sweet);

constraint forall (i in 1..len-1) (temp1[i] = ((bool2int(taste[dish[i]] != taste[dish[i+1]])) + (bool2int(temp[dish[i]] != temp[dish[i+1]])) + (bool2int(heavy[dish[i]] != heavy[dish[i+1]]))));
% After spicy must be either bland or sweet

% After sour must be bland or umami

% No spicy or umami after sweet

% Hot dish -> Warm dish -> Cold dish

constraint forall (i in 1..len-1, j in i+1..len) ((temp[dish[i]] = hot /\ temp[dish[j]] = cold) -> exists (k in i + 1..j -1)(temp[dish[k]] = warm));

% No two heavy dishes in a row 

% Maximize sum of value + count of changes in taste, temperature and weight between each pairs

var int : changes;

constraint changes = sum (i in 1..len-1) (temp1[i]);

constraint obj = sum (d in dish) (value[d]) + changes;
