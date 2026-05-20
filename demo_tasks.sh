#!/bin/bash
pause_secs=5
#do_teleport="-t"
#do_teleport_with_pause="-t --pause"
#no_teleport=""
teleport="-t"

num_loops=${1:-1} # defaults to 1 if command line arg 1 is not given
for ((i=1;i<=$num_loops;i++));
do

echo "Next task ... TableTask1"
read -t $pause_secs -p "  Pick 16 colored cubes (4 each of red, green, blue, orange) from the bin and arrange them in a tightly packed 4x4 grid on the conveyor with horizontal color stripes."
python run_task.py -x $teleport --task TableTask1


echo "Next task ... TableTask2"
read -t $pause_secs -p "  Pick multiple cubes and place them on blue cubes arranged in a 3x4 grid."
python run_task.py -x $teleport --task TableTask2


echo "Next task ... TableTask3"
read -t $pause_secs -p "  Pick balls from the bin and place them onto disc targets arranged in a 3x4 grid on the dropzone table."
python run_task.py -x $teleport --task TableTask3

echo "Next task ... TableTask3b"
read -t $pause_secs -p "  Pick balls from the bin and place them into gaps between disc targets arranged in a tight 3x4 grid on the dropzone table."
python run_task.py -x $teleport --task TableTask3b


echo "Next task ... TableTask3c"
read -t $pause_secs -p "  Pick balls from the bin into disc-gap pockets, then place red balls from the cart into gaps between the placed balls."
python run_task.py -x $teleport --task TableTask3c


echo "Next task ... TableTask4"
read -t $pause_secs -p "  Pick cubes from the bin and place them onto yellow rectangles arranged in a circle on the dropzone table."
python run_task.py -x $teleport --task TableTask4


echo "Next task ... TableTask5"
read -t $pause_secs -p "  Pick green cubes from the dropzone table and place them onto red rectangles arranged in a circle."
python run_task.py -x $teleport --task TableTask5


echo "Next task ... TableTaskBottles1"
read -t $pause_secs -p "  Pick bottles from the bin and place them into carrier pads in an Nx4 grid in the dropzone (N columns, 1-3)."
python run_task.py -x $teleport --task TableTaskBottles1


echo "Next task ... TableTaskBottlesToConveyor"
read -t $pause_secs -p "  Pick bottles from 2 stacked layers in the bin and place them into carrier pads in a row on the conveyor."
python run_task.py -x $teleport --task TableTaskBottlesToConveyor


echo "Next task ... TableTaskCrackerBoxes1"
read -t $pause_secs -p "  Pick cracker boxes from the bin and place them onto thin green rectangles arranged in a 3x4 grid in the dropzone."
python run_task.py -x $teleport --task TableTaskCrackerBoxes1 --target-count-min 8 --target-count-max 11


echo "Next task ... TableTaskCartToConveyor"
read -t $pause_secs -p "  Pick cracker boxes, soup cans, mustard bottles, and sugar boxes from the cart and place one of each vertically into boxes on the conveyor."
python run_task.py -x $teleport --task TableTaskCartToConveyor


echo "Next task ... TableTaskColorBinSort"
read -t $pause_secs -p "  Pick between one and five cubes and balls (each tinted red, green, or blue) from the conveyor and drop them into the color-matching collection boxes."
python run_task.py -x $teleport --task TableTaskColorBinSort


echo "Next task ... TableTaskColorCircle"
read -t $pause_secs -p "  Pick randomly colored cubes from the bin and place them in a circle on the drop zone."
python run_task.py -x $teleport --task TableTaskColorCircle


echo "Next task ... TableTaskColorShapes"
read -t $pause_secs -p "  Pick cubes, cylinders, cones, and balls spawned on the conveyor (each tinted red, green, or blue) and place them into the matching colored boxes on the table."
python run_task.py -x $teleport --task TableTaskColorShapes


echo "Next task ... TableTaskColors1"
read -t $pause_secs -p "  Pick colored cubes from the bin (4x3 grid of red/green/blue) and place them onto matching colored markers (3x4 grid including cyan/magenta) in the dropzone."
python run_task.py -x $teleport --task TableTaskColors1 --target-count-min 10 --target-count-max 12


echo "Next task ... TableTaskConveyorColorStacks"
read -t $pause_secs -p "  Pick colored cubes from the conveyor and stack them in the bin in triplets: blue (bottom), green (middle), red (top). Skip yellow cubes and excess cubes."
python run_task.py -x $teleport --task TableTaskConveyorColorStacks --seed 10


echo "Next task ... TableTaskConveyorSort"
read -t $pause_secs -p "  Pick cubes and balls from the conveyor and sort them into separate boxes on the cart."
python run_task.py -x $teleport --task TableTaskConveyorSort


echo "Next task ... TableTaskCrackerCircle"
read -t $pause_secs -p "  Pick horizontal cracker boxes from a circle and stack them in the pick bin on the cart."
python run_task.py -x $teleport --task TableTaskCrackerCircle --pick-count 7


echo "Next task ... TableTaskLayeredCircle"
read -t $pause_secs -p "  Pick horizontal sugar boxes from a circle stacked 2 layers high (10 total) and stack them in the pick bin on the cart."
python run_task.py -x $teleport --task TableTaskLayeredCircle --pick-count-min 4 --pick-count-max 10


echo "Next task ... TableTaskLayeredCubes"
read -t $pause_secs -p "  Pick cubes from a 2x3 grid stacked 3 layers high (18 total) and place them onto markers in the dropzone."
python run_task.py -x $teleport --task TableTaskLayeredCubes


echo "Next task ... TableTaskMixedCircle"
read -t $pause_secs -p "  Pick mixed items (cubes, cones, cylinders, bottles) arranged in a circle on the conveyor and place them into the bin on the cart."
python run_task.py -x $teleport --task TableTaskMixedCircle


echo "Next task ... TableTaskMixedPacking"
read -t $pause_secs -p "  Pick Cracker Boxes and Soup Cans from 5 rows on the conveyor (1 row Boxes, 4 rows Cans) and place four Soup Cans and one Cracker Box into each of two boxes on the cart. The Cans should form a 2x2 grid."
python run_task.py -x $teleport --task TableTaskMixedPacking --seed 14773
#--seed 35413 # FAILS during first pick (collision during transport)


echo "Next task ... TableTaskShapeSortBoxes"
read -t $pause_secs -p "  Pick randomly colored cubes and balls from the conveyor and sort them by shape into two boxes on the cart."
python run_task.py -x $teleport --task TableTaskShapeSortBoxes


# echo "Next task ... TableTaskSortAndStack"
# read -t $pause_secs -p "  Pick red, green, and blue cubes from a 6x5x3 stacked grid and sort them into matching color-coded boxes on the cart, stacking cubes on top of previously placed cubes."
# python run_task.py -x $teleport --task TableTaskSortAndStack


echo "Next task ... TableTaskSortAndStack2"
read -t $pause_secs -p "  Pick red, green, and blue cubes from a 6x5x3 stacked grid and sort them into matching color-coded boxes on the cart, stacking cubes on top of previously placed cubes. Yellow cubes are relocated to 6 stacks on the dropzone floor (to the right (+X) and closer to the robot (-Y) than the source pile)."
python run_task.py -x $teleport --task TableTaskSortAndStack2


echo "Next task ... TableTaskSoupCanPacking"
read -t $pause_secs -p "  Pick soup cans from the conveyor and place 6 into each of 4 boxes on the cart."
python run_task.py -x $teleport --task TableTaskSoupCanPacking


echo "Next task ... TableTaskSoupCans1"
read -t $pause_secs -p "  Pick soup cans from the bin and place them onto thin red rectangles arranged in a 3x4 grid in the dropzone."
python run_task.py -x $teleport --task TableTaskSoupCans1  --target-count-min 5 --target-count-max 7


done
