from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput

def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set("aeiouAEIOU")
    return sum(1 for char in text if char in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count('1')


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        count += 1
    return count


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    return sum((p - e) ** 2 for p, e in zip(predicted, expected)) / n


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = {}
    divisor = 2
    while divisor * divisor <= number:
        while number % divisor == 0:
            factors[divisor] = factors.get(divisor, 0) + 1
            number //= divisor
        divisor += 1
    if number > 1:
        factors[number] = factors.get(number, 0) + 1

    result = []
    for prime, power in sorted(factors.items()):
        if power == 1:
            result.append(f"({prime})")
        else:
            result.append(f"({prime}**{power})")
    return "".join(result)


def pyramid(data: PositiveIntegerInput):
    cube_count = data.value
    total = 0
    k = 1
    while total < cube_count:
        total += k * k
        if total == cube_count:
            return k
        k += 1
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    s = str(number)
    length = len(s)
    mid = length // 2
    
    if length % 2 == 0:
        left = sum(int(d) for d in s[:mid])
        right = sum(int(d) for d in s[mid:])
    else:
        left = sum(int(d) for d in s[:mid])
        right = sum(int(d) for d in s[mid+1:])
        
    return left == right