% nurse rostering project
% STAGE A data
enum NURSE; % set of nurses to roster
enum SHIFT = { MORN, DAY, EVEN, NIGH, OFF };
int: nday;     % number of days for roster
set of int: DAYS = 1..nday; 
array[NURSE,DAYS] of bool: rostered_off; % nurse is guaranteed an off shift
% STAGE B data
int: maxweek; % maximum working shifts in any 7 day period
int: maxnightfort; % maximum night shifts in any 14 day period
int: minfort; % minimum number of working shifts in any 14 day period
array[SHIFT,DAYS] of int: minshift; % minimum number of nurses rostered on each shift
set of int: DAYSPERWEEK = 0..6;
set of int: DAYSPERFORT = 0..13;
% STAGE C data
array[NURSE] of int: shift_cost;  
% STAGE D data
enum WARD;  % set of wards to roster
WARD: dummy; % dummy ward
array[WARD,DAYS] of int: minward; % minimum number of nurses rostered to each ward
constraint forall(d in DAYS)(assert(minward[dummy,d] = 0, "Dummy ward has non zero minimum on day \(d)\n"));
int: maxward; % maximum number of wards any nurse can work in the roster period
% STAGE E data
enum SKILL;
SKILL: senior;                         % which skill is senior
array[NURSE] of set of SKILL: skill;   % skills that each nurse has
array[WARD] of set of SKILL: required; % particular skills for each ward
opt WARD: emergency; % emergency ward if its being rostered
enum SLICE = { PREDAWN, AM, PM, LATE };
% coverage MORNING = {PREDAWN, AM}, 
%          DAY = {AM,PM} 
%          EVENING = {PM, LATE}, 
%          NIGHT {LATE, PREDAWN of the next day} 
set of SLICE: MORNING_SLICE = {PREDAWN, AM};
set of SLICE: DAY_SLICE  = {AM, PM};
set of SLICE: EVENING_SLICE  = {PM, LATE};
set of SLICE: NIGHT_SLICE  = {LATE, PREDAWN};

%%%% DECISIONS
array[NURSE,DAYS] of var SHIFT: sh; % shift for each nurse on each day
var 0..nday*sum(shift_cost): total_cost;
array[NURSE,DAYS] of var WARD: wd; % ward rostered for each nurse


include "globals.mzn";
include "regular.mzn";  

% (A2) Transition Table for A2
% MORN = no rules
% DAY = DAY1 -> DAY2 (Then DAY2 is unrestricted). This was to ensure Day is always followed by another Day
% EVEN = EVEN -> all except for MORN
% NIGH = NO NIGH -> DAY or MORN, NIGH1 transitions to NIGH2 and to NIGH3 after seeing three consecutive NIGHS to ensure we dont see another one
% OFF = no OFF -> OFF    (OFF1 will transition to OFF2 when OFF is observed, which cannot observe another OFF)  
array[1..10,SHIFT] of 0..10: trans =
      [|1,2,3,4,5  %MORN(1)
      | 0,8,0,0,0  %DAY(2)
      | 0,2,3,4,5  %EVEN(3)
      | 0,0,3,6,5  %NIGH1(4)
      | 1,2,3,4,9  %OFF1(5)
      | 0,0,3,7,5  %NIGH2(6)
      | 0,0,3,0,5  %NIGH3(7)
      | 1,8,3,4,5  %DAY2(8)
      | 1,2,3,4,0  %OFF2(9)
      | 1,2,3,4,5|]; %DUMMY(10)     

% (D) Loops across all shifts such that if their shifts are off then the ward assigned is a dummy ward. This constraint uses the biconditional operator
constraint forall (n in NURSE, day in DAYS) (sh[n,day] = OFF <-> wd[n,day] = dummy);

% (A1) Loops through all nurses that are rostered off from the given rostered_off and ensures that we do not assign any shifts to them
constraint forall (n in NURSE, day in DAYS) (rostered_off[n,day] -> (sh[n,day] = OFF));

% (A2) Applies the regular constraint according to the transition array mentioned above
constraint forall(n in NURSE) (regular ([sh[n,day] |day in DAYS],10,5,trans,10,{1,2,3,4,5,6,7,8,9,10}));

% (B1) Each Shift s for each d has at least minshift [s,d] (Global constraint used = global cardinality)
% Used global constraints to ensure solver defines the most efficient one to solve this problem
constraint forall (day in DAYS) 
  (global_cardinality (
    [sh[n,day] | n in NURSE],
    [MORN, DAY, EVEN, NIGH, OFF],
    [minshift[MORN,day],minshift[DAY,day],minshift[EVEN,day],minshift[NIGH,day],minshift[OFF,day]],
    [card(NURSE),card(NURSE),card(NURSE),card(NURSE),card(NURSE)]
));

% (B2) Auxiliary array to represents whether the shift assigned is off (used to prevent usage ot bool2int within array)
array [NURSE,DAYS] of var 0..1: sh_off;
array [NURSE,DAYS] of var 0..1: sh_nigh;

% (B2) Create auxiliary array to convert into boolean context for sliding sum
constraint forall (n in NURSE, day in DAYS) (sh_off[n,day] = bool2int(sh[n,day] != OFF) /\ sh_nigh[n,day] = bool2int(sh[n,day] = NIGH));

% (B2) Perform sliding sum on the array created to enforce constraints listed in B2
% To fulfill maxweek, we set the wrindow to 7 and loop through the sh_off array to ensure that for each of them they have to be within maxweek.
% To fulfill minfort, we set the window to 14 and loop through the sh_off array to ensure for each window they have to be between minfort and nday
% To fulfill maxnightfort, we set the window to 14 and loop through the sh_nigh array to ensure for each window they are within maxnightfort
constraint forall (n in NURSE) (sliding_sum(0, maxweek, 7,[sh_off[n, day] |day in DAYS]) /\ sliding_sum(minfort, nday, 14,[sh_off[n, day] | day in DAYS]) /\ sliding_sum(0, maxnightfort, 14,[sh_nigh[n, day] | day in DAYS]));

% (C) Calculate shift cost for all nurses across all days based on the sh_off array that we have created (if off then no cost, else add cost)
constraint total_cost = (sum(n in NURSE,d in DAYS) (sh_off[n,d]* shift_cost[n]));

% (D1) Nurses for that ward on that particular day
array[WARD,DAYS] of var int: count_wd; 

% (D1) Calculates the number of nurses assigned to the ward on that particular day
constraint forall (w in WARD, day in DAYS) (count_wd[w,day] = (sum(n in NURSE) (wd[n,day] = w)));

% (D1) Constraints count to be minward at least 
% Seperated from the other D1 constraint because performance is worse if combined within the same loop (even though they are looping through the same array)
constraint forall (w in WARD, day in DAYS) (count_wd[w,day] >= minward[w,day]);

% (D2) Distinct wards for maxward constraint
% Reasoning for using an auxiliary array is so that i can set the bounds to maxward as opposed to writing an additional loop to constraint that
array [NURSE] of var 1..maxward : distinct_wards;

% (D2) Calculate distinct ward using nvalue constraint
% For each distinct wards value (for each nurse) the value will be nvalue(distinct) of their assigned wards across the days
constraint forall (n in NURSE) (distinct_wards[n] = nvalue([wd[n,day]|day in DAYS]));

% Part E Incomplete implementation
% % If wd = ward then ward's speciality has to be on duty
% set of int: skill_max = 0..card(SKILL) +1;
% array [NURSE, DAYS] of var skill_max: testing;
% constraint forall(n in NURSE,day in DAYS)(testing[n,day] =  card(skill[n] intersect required[wd[n,day]]) + (if (required[wd[n,day]] = {}) then 1 else 0 endif));
% array [WARD,DAYS] of var 0..card(NURSE): testing1; 
% constraint forall (w in WARD, day in DAYS) (if (minward[w,day] > 0) then ((testing1[w,day] = (sum (n in NURSE) (bool2int(wd[n,day] = w)*testing[n,day]))) > 0) endif);
% For each slice, there must be a senior

% If emergency = True, 

solve minimize total_cost;