% train scheduling
%%%%%%%%%%% Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
enum STOP;                             % set of stops
STOP: dstop;                           % dummy stop
array[STOP] of int: minimal_wait;      % minimum wait time at station 
array[STOP,STOP] of opt 0..infinity: travel_time; % travel time from each stop to another <> if not possible
array[STOP] of int: platform;          % number of platforms at station
enum STYPE = { ORDINARY, HUB, TERMINUS }; % Ordinary = normal, Hub = where lines meet, Terminus = where services can begin and end
array[STOP] of STYPE: stype;           % type of STOP
array[STOP] of int: skip_cost;         % cost to skip the station (per service)
enum LINE = { SING, DOUB, QUAD, NONE };  
% SING = Single track for both directions (need to have one train leave the connection before starting a train in opposite direction, 
% DOUB = Single track for each of both direction (two directions are independant (like normal train tracks) (2 tracks in total)
% QUAD = Doubel trackes for both direction (4 Tracks in total)

array[STOP,STOP] of LINE: line;        % line type between stops
                  
enum ROUTE;                            % a sequence of STOPs
int: max_route_length;                 % longest route in number of stops
set of int: STOPNO = 1..max_route_length;         
array[ROUTE] of STOPNO: rlength;       % length of route
array[ROUTE,STOPNO] of STOP: route;    % route definitions; 
   
enum SERVICE;                          % the set of services to be schedules                         
array[SERVICE] of ROUTE: sroute;       % the route for each service
int: makespan;                         % end of time considered;
set of int: TIME = 0..makespan;
TIME: min_sep;                         % min separation of services on the same line same direction
array[SERVICE] of TIME: service_start; % from when the service can start
array[SERVICE] of TIME: service_end;   % when the service ideally ends

enum ENGINE;                           % set of engines available
array[ENGINE] of STOP: start;          % start location for engines
enum SERVICEX = s(SERVICE) ++ e(ENGINE); 
%%%%%%%%%%% Decisions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

array[SERVICE,STOPNO] of var TIME: arrive;    % arrive time at each stop
array[SERVICE,STOPNO] of var TIME: depart;    % depart time at each stop
array[SERVICE,STOPNO] of var TIME: wait;      % wait time at each stop
array[SERVICE,STOPNO] of var bool: stopped;   % which stops are actually made                                                            
var 0..infinity: delay_obj;
var 0..infinity: skip_obj; 
array[SERVICE] of var ENGINE: engine;         % which engine is used for each service
array[SERVICE] of var SERVICEX: prev;         % previous service or engine

% Stage A: Schedule Constraints
% No service starts (arrives) at first station before service start time
% Wait time for each stop is depart time - arrive time
% At every stop, wait time is >= minimal wait time
% Arrival time of next stop is >= travel time from prev stop + departure time of prev stop
% Dummy stop =  wait time = 0, (basically departure and arrival =  both departure time from previous stop)

% Stage A: No service starts (arrives) at first station before service start time
constraint forall (ser in SERVICE, stop in STOPNO) (arrive[ser,stop] >= service_start[ser]);
% Stage A: Wait time for each stop is depart time - arrive time
constraint forall(ser in SERVICE, stop in STOPNO) (wait[ser,stop] = depart[ser,stop] - arrive[ser,stop] /\ service_stops[ser,stop] = route[sroute[ser],stop]);
% Stage A: Intermediate array to access specific stop using service and stopno (to improve efficiency)
array[SERVICE, STOPNO] of var STOP: service_stops;
% Stage A: At every stop, wait time is >= minimal wait time
constraint forall(ser in SERVICE, stop in STOPNO) (service_stops[ser,stop] = route[sroute[ser],stop]);
constraint forall (ser in SERVICE, stop in STOPNO) (wait[ser,stop] >= minimal_wait[service_stops[ser,stop]]);
% Stage A: Arrival time of next stop is >= travel time from prev stop + departure time of prev stop
constraint forall (ser in SERVICE, stop in 1..max_route_length - 1) (arrive[ser,stop+1] >= depart[ser,stop] + travel_time[service_stops[ser,stop],service_stops[ser,stop+1]]);
% Stage A: Dummy stop =  wait time = 0, (basically departure and arrival =  both departure time from previous stop)
% Starts at two to avoid array accesssing 0 error (when dstop is the first stop in the route)
constraint forall (ser in SERVICE, stop in 2..max_route_length where service_stops[ser,stop] == dstop) (arrive[ser,stop] == depart[ser, stop-1]);
% Sets depart time to be equal to previous stop
constraint forall (ser in SERVICE, stop in 1..max_route_length where service_stops[ser,stop] == dstop) (depart[ser, stop] == arrive[ser,stop]);


                                                                                    
