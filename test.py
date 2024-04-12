import re
from collections import defaultdict

def refine_ini_obs(ini_obs):
    head, obs = ini_obs.split('. ')
    items_with_numbers = re.findall(r'(\w+)\s(\d+)', obs)

    item_ranges = defaultdict(list)
    for item, number in items_with_numbers:
        item_ranges[item].append(int(number))
        
    formatted_items = []
    for item, numbers in item_ranges.items():
        numbers.sort()
        if len(numbers) == 1:
            formatted_items.append(f"{item} ({numbers[0]})")
        else:
            formatted_items.append(f"{item} ({numbers[0]}-{numbers[-1]})")

    formatted_text = ', '.join(formatted_items)
    return head + ". Looking quickly around you, you can see " + formatted_text


ini_obs = "You are in the middle of a room. Looking quickly around you, you see a bed 1, a sidetable 1, a drawer 1, a dresser 1, a drawer 2, a drawer 3, a drawer 4, a drawer 5, a drawer 6, a drawer 7, a drawer 8, a drawer 9, a drawer 10, a drawer 11, a safe 1, a laundryhamper 1, and a garbagecan 1."

print(refine_ini_obs(ini_obs))