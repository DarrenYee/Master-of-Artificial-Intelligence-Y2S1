% -- Part 1 --

% 1.1 --- Multiples of Three
array[int] of int: multiples_of_three ::output = [ /* TODO: insert answer */ ];

% 1.2 --- Even Multiples of Three
array[int] of int: even_multiples_of_three ::output = [ /* TODO: insert answer */ ];

% 1.3 --- Products
array[int] of int: products ::output = [ /* TODO: insert answer */ ];

array[int] of int: products_div_3_not_2 ::output = [ /* TODO: insert answer */ ];

% -- Part 2 --

% 2.1 --- Manhattan Distance

set of int: Point = 1..5;
array[Point] of int: x_coord = [1,3,5,1,10];
array[Point] of int: y_coord = [4,3,8,1,7];

%% uncomment the following array when you start building an answer
%% otherwise we get an Index set mismatch since the rhs is a 1D array
%array[Point,Point] of int: manhattan_distance ::output = [ /* TODO: insert answer */ ];

% 2.2 --- Manhattan Distance with Dummy

set of int: Point0 = 0..5;
%% uncomment when you start building the answer, similarly for this and the 2D arrays in 2.3
%array[Point0,Point0] of int: manhattan_distance0 ::output = [ /* TODO: insert answer */ ];

% 2.3 --- Euclidian Distance with Dummy

%array[Point,Point] of float: euclidian_distance = [ /* TODO: insert answer */ ];

%array[Point0,Point0] of float: euclidian_distance0 ::output = [ /* TODO: insert answer */ ];

% -- Part 3 --

% 3.1  --- All Different

int: n = 5;
array[1..n] of var 1..n: x ::output;

% TODO: Enforce all_different any way you can think of

% 3.2 --- All Different with index_set(x)

% TODO: Enforce all_different using index_set(x)

% -- Part 4 --

% 4.1 --- Output Even Multiples of Three

% Define your output statement in this declaration
array[int] of string: output_step1 ::output_only = [ /* TODO: insert answer */ ]; 

% Uncomment the next link to try your output statement
% output output_step1;

% Output string to give to the checker
string: step1_checker ::output = concat(output_step1);

% 4.2 --- Output Manhattan Distance

% Define your output statement in this declaration
array[int] of string: output_step2 ::output_only =  [ /* TODO: insert answer */ ];

% Uncomment the next link to try your output statement
% output output_step2;

% Output string to give to the checker
string: step2_checker ::output = concat(output_step2);
