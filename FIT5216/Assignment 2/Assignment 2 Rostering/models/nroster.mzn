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

%%%% DECISIONS
array[NURSE,DAYS] of var SHIFT: sh; % shift for each nurse on each day
var 0..nday*sum(shift_cost): total_cost;
array[NURSE,DAYS] of var WARD: wd; % ward rostered for each nurse

include "regular.mzn";  

      
array[1..8,SHIFT] of 0..8: d =
      [|1,2,3,4,5  %MORN
      | 0,8,0,0,0  %DAY
      | 0,2,3,4,5  %EVEN
      | 0,0,3,6,5  %NIGH1
      | 1,2,3,4,5  %OFF1
      | 0,0,3,7,5  %NIGH2
      | 0,0,3,0,5  %NIGH3
      | 1,8,3,4,5|];  %DAY2     
       
constraint forall (i in NURSE, j in DAYS) (rostered_off[i,j] -> sh[i,j] = OFF);

constraint regular ([sh[i,j] | i in NURSE, j in DAYS],8,5,d,1,SHIFT);

constraint forall (i in NURSE, j in 1..nday - 2) (sh[i,j] = OFF /\ sh[i,j+1] = OFF -> sh[i,j+2] != OFF);

% Each Shift s for each d has at least minshift [s,d]
% No more than maxweek shifts (excluding off) in 7 day period
% No less than minfort in 14 day period
% No more than maxnightfort NIGH in any 14 day perdod

include "globals.mzn";

constraint forall (day in DAYS) 
  (global_cardinality (
    [sh[n,day] | n in NURSE],
    [MORN, DAY, EVEN, NIGH, OFF],
    [minshift[MORN,day],minshift[DAY,day],minshift[EVEN,day],minshift[NIGH,day],minshift[OFF,day]],
    [999999,999999,999999,999999,999999]
));


constraint forall (week in 1..nday div 7)
  (global_cardinality(
    [sh[n,day] | n in NURSE, day in (week*7)..week+7],
  [],
  []
))


