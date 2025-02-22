def normalize_output(output):
    """
    In order to eliminate ambiguous and minor variations in the output, we will normalize the output.
    Convert the given string to a standardized canonical form by:
    1) Converting written-out numbers to digits.
    2) Mapping synonyms to a single common term.
    3) Converting plural forms to singular.
    """

    # Map written-out numbers to digit strings
    number_mapping = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20',
    }

    # Map known synonyms to a single canonical term
    synonym_mapping = {
        'refrigerator': 'fridge',
        "refrigerators": "fridge",
        'stove': 'oven',
        'alligator': 'crocodile',
        'porpoise': 'dolphin',
        'automobile': 'car',
        'nyc': 'new york city',
        'la': 'los angeles',
        'usa': 'united states',
        'co2': 'carbon dioxide',
        'o2': 'oxygen',
        'n2': 'nitrogen',
        'h2o': 'water',
        'tortoise': 'turtle',
        'motorbike': 'motorcycle',
        'cellphone': 'phone',
        'telephone': 'phone',
        'pc': 'computer',
        'tv': 'television',
        'tap': 'faucet',
        'aeroplane': 'airplane',
        'cubic': 'cube',
        'cubical': 'cube',
        'cubes': 'cube',
        'cuboids': 'cube',
        'cuboid': 'cube',
        'square': 'cube',
        'squares': 'cube',
        'striped': 'stripes',
        'checkered': 'checkerboard',
        'polka-dots': 'spots',
        'dalmatian': 'dog',
        'triangular': 'triangle',
        'circular': 'round',
        'circle': 'round',
        'circles': 'round',
        'spherical': 'round',
        'spheres': 'round',
        "sphere": "round",
        "roundpherical": "round",
        "roundph": "round",
        'triangles': 'triangle',
        'logs': 'wood',
        'zigzag': 'curved',
        'hexagonal': 'hexagon',  
        'bud': 'flower',
        'hippopotamus': 'hippo',
        'rhinoceros': 'rhino',
        'bike': 'bicycle',
        'schoolbus': 'bus',
        'boat': 'ship',
        "boats": "ship",
        'sailboat': 'ship',
        'airship': 'ship',
        'donut': 'torus',
        'donuts': 'torus',
        'wallaby': 'kangaroo',
        'teacup': 'cup',         
        'teapot': 'kettle',
        'rooster': 'chicken',
        'roosters': 'chicken',
        'raven': 'crow',
        'vineyard': 'vine',
        'bushe': 'bush',
        'crystal': 'glass',
        'hay': 'straw',
        'fireplace': 'oven',
        'co₂': 'carbondioxide',
        "carbon dioxide": "carbondioxide",
        "beesap": "bee",
        'aircondition': 'AC',
        'airconditioner': 'AC',
        'air-conditioner': 'AC',
        't-rex': 'dinosaur',
        'trex': 'dinosaur',
        'man': 'person',
        'woman': 'person',
        'people': 'person',
        'men': 'person',
        'women': 'person',
        'clocktower': 'bigben',
        'multicolored': 'rainbow',
        'thatch': 'straw',
        'plane': 'airplane',
        'goggles': 'glasses',
        'night-vision': 'glasses',
        'blossoms': 'flower',
        'brush': 'eraser',
        'serpent': 'snake',
        'dots': 'spots',
        'binoculars': 'glasses',
        'slippers': 'shoe',
        'slipper': 'shoe',
        'pillow': 'cushion',
        'hexagons': 'hexagon',
        'ukulele': 'guitar',
        'cello': 'violin',
        'America': 'USA',
        'steel': 'metal',
        'cucumber': 'pickle',
        'galaxy': 'space',
        'underwater': 'sea',    
        'ocean': 'sea',
        'faceted': 'diamond',
        'jewelry': 'diamond',
        'jewelries': 'diamond',
        'backpack': 'bag',
        'squid': 'octopus',
        'kitten': 'cat',
        'octagonal': 'octagon',
        'candy': 'lolipop',
        'pipeline': 'pipe',
        'dragonfruit': 'pitaya',
        "new york": "new york city",
        "eyesight": "eye",
        "seismograph": "seismometer",
    }

    # Convert plural forms to singular
    plural_singular_mapping = {
        'butterflies': 'butterfly',
        'bees': 'bee',
        'ants': 'ant',
        'wasps': 'wasp',
        'kangaroos': 'kangaroo',
        'koalas': 'koala',
        'wombats': 'wombat',
        'trees': 'tree',
        'books': 'book',
        'goats': 'goat',
        'squirrels': 'squirrel',
        'rabbits': 'rabbit',
        'pandas': 'panda',
        'giraffes': 'giraffe',
        'lions': 'lion',
        'tigers': 'tiger',
        'cows': 'cow',
        'horses': 'horse',
        'cats': 'cat',
        'dogs': 'dog',
        'whales': 'whale',
        'sharks': 'shark',
        'dolphins': 'dolphin',
        'flowers': 'flower',
        'leaves': 'leaf',
        'knives': 'knife',
        'wolves': 'wolf',
        'mice': 'mouse',
        'geese': 'goose',
        'children': 'child',
        'teeth': 'tooth',
        'feet': 'foot',
        'fungi': 'fungus',
        'stimuli': 'stimulus',
        'media': 'medium',
        'octopi': 'octopus',
        'cacti': 'cactus',
        'diamonds': 'diamond',
        'bricks': 'brick',
        'flame': 'fire',
        'winds': 'wind',
        'wheels': 'wheel',
        'chickens': 'chicken',
        'fireflies': 'firefly',
        'beaks': 'beak',
        'needles': 'needle',
        'spinners': 'spinner',
        'clouds': 'cloud',
        'earthquakes': 'earthquake',
        'seals': 'seal',
        'pencils': 'pencil',
        'petals': 'petal',
        'forks': 'fork',
        'seahorses': 'seahorse',
        'keys': 'key',
        'carrots': 'carrot',
        'crayons': 'crayon',
        'skyscrapers': 'skyscraper',
        'birds': 'bird',
        'bicycles': 'bicycle',
        'watches': 'watch',
        'lemons': 'lemon',
        'pipes': 'pipe',
        'spinnerets': 'spinneret',
        'bubbles': 'bubble',
        'camels': 'camel',
        'stripes': 'stripe',
        'lungs': 'lung',
        'gills': 'gill',
        'feathers': 'feather',
        'scales': 'scale',
        'lollipops': 'lolipop',
        'lollipop': 'lolipop',
        'lolipops': 'lolipop',
        'drums': 'drum',
        'ropes': 'rope',
        'shoes': 'shoe',
        "bushes": "bush",
        "elephants": "elephant",
        "porcupines": "porcupine",
        "clocks": "clock",
        "antelopes": "antelope",
        "eyes": "eye",
        "chameleons": "chameleon",
        "rockets": "rocket",
        "turbines": "turbine",
        "ostriches": "ostrich",
        "pumpkins": "pumpkin",
        "shrubs": "shrub",
        "fields": "field",
    }

    # Preprocess the output: lowercase and strip trailing whitespace/punctuation
    output = str(output).lower().strip()
    if output.endswith('.'):
        output = output[:-1]  # remove trailing period if present

    # Apply the three mappings in sequence
    output = number_mapping.get(output, output)
    output = synonym_mapping.get(output, output)
    output = plural_singular_mapping.get(output, output)

    return output


def compare_single_output(our, gt):
    """
    Compare a single predicted (our) string to the ground truth (gt)
    after normalizing both. Return True if they match exactly; otherwise False.
    """
    our_norm = normalize_output(our)
    gt_norm = normalize_output(gt)

    # Return True only if normalized outputs match and are not 'none'
    if our_norm == gt_norm and our_norm != 'none':
        return True
    return False


def compare_outputs(our_results, gt_results):
    """
    Compare two lists of outputs (ours vs. ground truth).
    Returns:
      1) Number of matches,
      2) Total comparisons,
      3) List of boolean match results for each pair.
    If the lists differ in length, a warning is printed and partial comparison is done.
    """

    if len(our_results) != len(gt_results):
        print("Warning: Lists have different lengths.")
        return 0, len(our_results), len(gt_results)

    matches = 0
    matches_details = []
    total = len(our_results)

    for our, gt in zip(our_results, gt_results):
        if compare_single_output(our, gt):
            matches += 1
            matches_details.append(True)
        else:
            matches_details.append(False)

    return matches, total, matches_details

