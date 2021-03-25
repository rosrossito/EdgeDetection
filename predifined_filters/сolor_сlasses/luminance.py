import enum

from сolor_сlasses.colors import Colors


class Luminance(enum.Enum):
    yellow_green = 0.299 * Colors.yellow_green.value[0] + 0.587 * Colors.yellow_green.value[1] + 0.114 * Colors.yellow_green.value[2]
    dark_olive_green = 0.299 * Colors.dark_olive_green.value[0] + 0.587 * Colors.dark_olive_green.value[1] + 0.114 * Colors.dark_olive_green.value[2]
    olive_drab = 0.299 * Colors.olive_drab.value[0] + 0.587 * Colors.olive_drab.value[1] + 0.114 * Colors.olive_drab.value[2]
    lawn_green = 0.299 * Colors.lawn_green.value[0] + 0.587 * Colors.lawn_green.value[1] + 0.114 * Colors.lawn_green.value[2]
    green_yellow = 0.299 * Colors.green_yellow.value[0] + 0.587 * Colors.green_yellow.value[1] + 0.114 * Colors.green_yellow.value[2]
    dark_green = 0.299 * Colors.dark_green.value[0] + 0.587 * Colors.dark_green.value[1] + 0.114 * Colors.dark_green.value[2]
    green = 0.299 * Colors.green.value[0] + 0.587 * Colors.green.value[1] + 0.114 * Colors.green.value[2]
    forest_green = 0.299 * Colors.forest_green.value[0] + 0.587 * Colors.forest_green.value[1] + 0.114 * Colors.forest_green.value[2]
    lime = 0.299 * Colors.lime.value[0] + 0.587 * Colors.lime.value[1] + 0.114 * Colors.lime.value[2]
    lime_green = 0.299 * Colors.lime_green.value[0] + 0.587 * Colors.lime_green.value[1] + 0.114 * Colors.lime_green.value[2]
    light_green = 0.299 * Colors.light_green.value[0] + 0.587 * Colors.light_green.value[1] + 0.114 * Colors.light_green.value[2]
    pale_green = 0.299 * Colors.pale_green.value[0] + 0.587 * Colors.pale_green.value[1] + 0.114 * Colors.pale_green.value[2]
    dark_sea_green = 0.299 * Colors.dark_sea_green.value[0] + 0.587 * Colors.dark_sea_green.value[1] + 0.114 * Colors.dark_sea_green.value[2]
    medium_spring_green = 0.299 * Colors.medium_spring_green.value[0] + 0.587 * Colors.medium_spring_green.value[1] + 0.114 * Colors.medium_spring_green.value[2]
    spring_green = 0.299 * Colors.spring_green.value[0] + 0.587 * Colors.spring_green.value[1] + 0.114 * Colors.spring_green.value[2]
    sea_green = 0.299 * Colors.sea_green.value[0] + 0.587 * Colors.sea_green.value[1] + 0.114 * Colors.sea_green.value[2]
    medium_aqua_marine = 0.299 * Colors.medium_aqua_marine.value[0] + 0.587 * Colors.medium_aqua_marine.value[1] + 0.114 * Colors.medium_aqua_marine.value[2]
    medium_sea_green = 0.299 * Colors.medium_sea_green.value[0] + 0.587 * Colors.medium_sea_green.value[1] + 0.114 * Colors.medium_sea_green.value[2]
    light_sea_green = 0.299 * Colors.light_sea_green.value[0] + 0.587 * Colors.light_sea_green.value[1] + 0.114 * Colors.light_sea_green.value[2]
    dark_slate_gray = 0.299 * Colors.dark_slate_gray.value[0] + 0.587 * Colors.dark_slate_gray.value[1] + 0.114 * Colors.dark_slate_gray.value[2]
    teal = 0.299 * Colors.teal.value[0] + 0.587 * Colors.teal.value[1] + 0.114 * Colors.teal.value[2]
    aqua = 0.299 * Colors.aqua.value[0] + 0.587 * Colors.aqua.value[1] + 0.114 * Colors.aqua.value[2]
    light_cyan = 0.299 * Colors.light_cyan.value[0] + 0.587 * Colors.light_cyan.value[1] + 0.114 * Colors.light_cyan.value[2]
    dark_turquoise = 0.299 * Colors.dark_turquoise.value[0] + 0.587 * Colors.dark_turquoise.value[1] + 0.114 * Colors.dark_turquoise.value[2]
    turquoise = 0.299 * Colors.turquoise.value[0] + 0.587 * Colors.turquoise.value[1] + 0.114 * Colors.turquoise.value[2]
    medium_turquoise = 0.299 * Colors.medium_turquoise.value[0] + 0.587 * Colors.medium_turquoise.value[1] + 0.114 * Colors.medium_turquoise.value[2]
    pale_turquoise = 0.299 * Colors.pale_turquoise.value[0] + 0.587 * Colors.pale_turquoise.value[1] + 0.114 * Colors.pale_turquoise.value[2]
    aqua_marine = 0.299 * Colors.aqua_marine.value[0] + 0.587 * Colors.aqua_marine.value[1] + 0.114 * Colors.aqua_marine.value[2]
