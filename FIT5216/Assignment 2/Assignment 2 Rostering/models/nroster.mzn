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

include "globals.mzn";

constraint forall (i in NURSE, j in DAYS) (rostered_off[i,j] -> sh[i,j] = OFF);

constraint forall (i in NURSE, j in 1..nday - 1) (sh[i,j] = NIGH -> sh[i,j+1] != MORN /\ sh[i,j] = NIGH -> sh[i,j+1] != DAY /\ sh[i,j] = EVEN -> sh[i,j+1] != MORN);

constraint forall (i in NURSE, j in 1..nday - 2) (sh[i,j] = OFF /\ sh[i,j+1] = OFF -> sh[i,j+2] != OFF);
constraint forall (i in NURSE) ((exists (j in DAYS) (sh[i,j] = DAY)) -> sum (k in 1..nday - 1) (sh[i,k] = DAY /\ sh[i,k+1] = DAY) <= 2);

% No nurse works more than 3 NIGH shifts in a row
% No nurse works a MORN shift directly after a NIGH shift
% No nurse works a DAY shift directly after a NIGH shift
% No nurse works a MORN shift directly after an EVEN shift
% No nurse has more than 2 OFF shifts in a row
% No nurse that works a DAY shift has less than 2 DAY shifts in a row (except the last shift
%can be a lone DAY shift)

