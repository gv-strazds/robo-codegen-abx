# TableTaskConveyorColorStacks2

## User Request
Pick colored cubes from the conveyor and stack them in the pick bin in stacks of 3: red on top of green on top of blue. Skip yellow cubes.

## Task Overview
- **Source**: 5-10 cubes on the conveyor, randomly colored red/green/blue/yellow
- **Target**: Existing pick bin on the table
- **Strategy**: ColorStackStrategy (custom color-ordered stacking)
- **Asset types**: cube (primitive)

## Task Description
Pick colored cubes from the conveyor and stack them in the bin in triplets: blue (bottom), green (middle), red (top). Yellow cubes and cubes that cannot form a complete triplet are skipped.

## Pick Items
- **Count**: 5-10 (random, seed-dependent)
- **Type**: cube
- **Colors**: random choice from red, green, blue, yellow
- **Position**: ConveyorPositionGenerator along Y-axis on the conveyor surface
- **Scale**: 0.0515 uniform

## Target Objects
- **Count**: 9 (3 stack positions x 3 layers)
- **Type**: hidden rect markers
- **Layout**: 3 stack positions spaced 0.10m along Y in the bin; each position has 3 vertical layers spaced by cube height (0.0515m)
- **Layer colors**: blue (bottom), green (middle), red (top)

## Pairing and Sequencing
- ColorStackStrategy analyzes pick colors and forms complete (blue, green, red) triplets
- Number of stacks = min(count_red, count_green, count_blue, 3)
- Picking order: within each stack, blue first (bottom), then green (middle), then red (top)
- Yellow cubes and excess cubes are not paired and not picked

## Success Condition
All cubes that were paired to targets are placed on their corresponding target markers.

## Success Checks
- Each paired cube rests on its assigned layer target (is_on_top check)
- Unpaired cubes (yellow, excess) remain on the conveyor
