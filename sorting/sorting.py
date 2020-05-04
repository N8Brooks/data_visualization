#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:05:50 2020

@author: nathan
"""

from itertools import permutations
from random import shuffle, sample, randrange
from bisect import bisect, bisect_left
from functools import total_ordering
from heapq import merge
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib import animation

def count_frames(func, arr):
    """
    function to count how many updates there are for a given sorting function
    """
    count = 1
    n = len(arr)
    pre = arr[:]
    for update in func(arr):
        if update != pre:
            pre = update[:]
            assert n == len(set(update))
            count += 1
            if count > 100000:
                return float('inf')
    return count

def display_plot(func, arr=None):
    """
    Call with specific function to show sorting function with an ipynb
    """
    if arr is None:
        arr = sample(range(1,129), 128)
    n = len(arr)
    remember = Counter(arr)
    domain = range(n)
    plt.axis('off')
    plt.bar(domain, arr, width=1., edgecolor='black')
    plt.show()
    
    pre = arr[:]
    for update in func(arr):
        if update != pre:
            pre = update[:]
            assert remember == Counter(update)
            plt.axis('off')
            plt.bar(domain, update, width=1., edgecolor='black')
            plt.show()
    
    assert all(update[i - 1] <= update[i] for i in range(1, n))

def animate_plot(func, arr=sample(range(1,129), 128)):
    """
    Call with specific function to download a mp4 of a sorting function
    """
    frames = [arr.copy()]
    n = Counter(arr)
    for update in func(arr):
        assert n == Counter(update)
        if update != frames[-1]:
            frames.append(update[:])
    
    assert all(frames[-1][i - 1] <= frames[-1][i] for i in range(1, n))
    
    # aim for 15 seconds with 60 fps
    frames = frames[::len(frames) // 900] + [frames[-1]]
    fcount = len(frames)
    
    fig = plt.figure(figsize=(19.2, 10.8))
    plt.axis('off')
    plt.tight_layout()
    barcollection = plt.bar(range(n), frames[0], width=1., edgecolor='black')
    
    def animate(i):
        y = frames[i]
        for i, b in enumerate(barcollection):
            b.set_height(y[i])
    
    anim=animation.FuncAnimation(fig, animate, blit=False, \
                                 frames=fcount, interval=1)
    
    anim.save(f'./{n}/{func.__name__}_{n}.mp4',\
              writer=animation.FFMpegWriter(fps=60))

def fisher_yates(arr):
    """
    modern fisher yates ascending shuffling method
    O(n)
    """
    n = len(arr)
    for i in range(n - 1):
        j = randrange(i, n)
        if i != j:
            arr[i], arr[j] = arr[j], arr[i]
            yield arr

def perm_sort(arr):
    """
    permutations sort
    worst: O(n!), average: O(n!/2), best: O(n)
    """
    length = len(arr)
    for perm in permutations(arr, length):
        yield perm
        if all(perm[j - 1] < perm[j] for j in range(1, length)):
            break

def bogo_sort(arr):
    """
    bogo sort or random sort
    worst: inf, average: O(n!), best: O(n)
    """
    length = range(1, len(arr))
    while any(arr[i] < arr[i - 1] for i in length):
        shuffle(arr)
        yield arr

def bubble_sort(arr):
    """
    bubble sort
    O(n^2)
    """
    length = len(arr)
    for i, x in enumerate(arr):
        for j in range(i + 1, length):
            if x > arr[j]:
                x, arr[j] = arr[j], x
                arr[i] = x
                yield arr

def cocktail_sort(arr):
    """
    cocktail sort
    O(n^2)
    """
    static = False
    n = len(arr)
    while not static:
        static = True
        for i in range(n - 2):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                static = False
                yield arr
        if static:
            break
        static = True
        for i in range(n - 2, -1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                yield arr
                static = False

def comb_sort(arr):
    """
    comb sort
    O(n^2)
    """
    gap = n = len(arr)
    swapped = True
    while gap != 1 or swapped:
        gap = gap * 10 // 13
        if gap < 1: gap = 1
        swapped = False
        for i in range(n - gap):
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                swapped = True
                yield arr
    
def islow_sort(arr, lo = None, hi = None):
    """
    slow sort
    O(n^(log2(n) / 2 + e))
    """
    if lo is None or hi is None:
        lo, hi = 0, len(arr) - 1
    if lo >= hi:
        return
    
    mi = (lo + hi) // 2
    for sub_arr in slow_sort(arr, lo, mi):
        yield sub_arr
    for sub_arr in slow_sort(arr, mi + 1, hi):
        yield sub_arr
    if arr[hi] < arr[mi]:
        arr[hi], arr[mi] = arr[mi], arr[hi]
        yield arr
    for sub_arr in slow_sort(arr, lo, hi - 1):
        yield sub_arr

def slow_sort(arr):
    """
    iterative slow sort with short circuiting
    O(n^2)
    """
    list1 = [(0, len(arr) - 1,)]
    list2 = list()
    
    while list1 or list2:
        while list1:
            lo, hi = list1.pop()
            if lo >= hi:
                continue
            mi = (lo + hi) // 2
            if all(arr[i] <= arr[i + 1] for i in range(lo, hi)):
                continue
            list1.extend(((lo, mi,), (mi + 1, hi,),))
            list2.append((lo, hi,))
        while list2:
            lo, hi = list2.pop()
            if lo >= hi:
                continue
            mi = (lo + hi) // 2
            if arr[hi] < arr[mi]:
                arr[hi], arr[mi] = arr[mi], arr[hi]
                yield arr
            list1.append((lo, hi - 1,))
            break

def insertion_sort(arr, lo = None, hi = None):
    """
    insertion sort
    O(n^2)
    """
    if lo is None or hi is None:
        lo, hi = 0, len(arr)
    for i in range(lo + 1, hi):
        j = i - 1
        key = arr[i]
        while j >= lo and key < arr[j]:
            arr[j + 1], arr[j] = arr[j], arr[j + 1]
            yield arr
            j -= 1

def binsertion_sort(arr, lo = None, hi = None):
    """
    binary insertion sort
    O(n^2)
    """
    if lo is None or hi is None:
        lo, hi = 0, len(arr)
    for i in range(lo + 1, hi):
        if arr[i] < arr[i - 1]:
            j = bisect(arr[lo:i], arr[i]) + lo
            arr.insert(j, arr.pop(i))
            yield arr

def patience_sort(arr):
    """
    patience sort
    worst: O(n^2), best: O(n)
    """
    @total_ordering
    class Pile(list):
        def __lt__(self, other): return self[-1] < other[-1]
        def __eq__(self, other): return self[-1] == other[-1]

    piles = []
    # sort into piles
    for j, x in enumerate(arr):
        new_pile = Pile([x])
        i = bisect_left(piles, new_pile)
        if i != len(piles):
            piles[i].append(x)
        else:
            piles.append(new_pile)
        yield list(merge(*[reversed(pile) for pile in piles])) + arr[j + 1:]
 
def strand_sort(a):
    """
    strand sort
    note: this is really poorly written
    worst: O(n^2), best: O(n)
    """
    i, out = 0, [a.pop(0)]
    while i < len(a):
        if a[i] > out[-1]:
            out.append(a.pop(i))
            yield out + a
        else:
            i += 1
    while len(a):
        i, tmp = 0, [a.pop(0)]
        while i < len(a):
            if a[i] > tmp[-1]:
                tmp.append(a.pop(i))
                yield out + tmp + a
            else:
                i += 1
            
        tmp2 = list()
        while out and tmp:
            if out[0] < tmp[0]:
                tmp2.append(out.pop(0))
            else:
                tmp2.append(tmp.pop(0))
            yield tmp2 + out + tmp + a
        while out:
            tmp2.append(out.pop(0))
            yield tmp2 + out + a
        while tmp:
            tmp2.append(tmp.pop(0))
            yield tmp2 + tmp + a
        out = tmp2

def selection_sort(arr):
    """
    selection sort
    O(n^2)
    """
    length = len(arr)
    for i, x in enumerate(arr):
        j = min(range(i, length), key=lambda j: arr[j])
        arr[i], arr[j] = arr[j], x
        yield arr

def shell_sort(arr): 
    """
    shell sort
    O(n^2)
    """
    length = len(arr)
    gap = length // 2
    while gap > 0:
        for i in range(gap, length):
            x = arr[i]
            while i >= gap and arr[i - gap] > x:
                arr[i], arr[i - gap] = arr[i - gap], arr[i]
                i -= gap
                yield arr
        gap //= 2

def odd_even_sort(arr):
    """
    odd even sorting algorithm
    O(n^2)
    """
    working = True
    while working:
        working = False
        for i in range(1, len(arr) - 2, 2):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                working = True
                yield arr
        for i in range(0, len(arr) - 1, 2):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                working = True
                yield arr

def spaghetti_sort(arr):
    """
    spaghetti sorting algorithm
    O(n^2), algthough if you are using physical spaghetti, it is O(n)
    """
    for hi in range(len(arr) - 1, 0, -1):
        imax = max(range(hi + 1), key=lambda i: arr[i])
        if imax != hi:
            arr[hi], arr[imax] = arr[imax], arr[hi]
            yield arr

def bitonic_sort(arr, lo=None, cnt=None, up=None):
    """
    bitonic sorting
    O(log^2(n))
    length of arr must be a base 2 number: 2, 4, 8, ...
    """
    def bitonic_merge(lo, cnt, up):
        if cnt > 1:
            k = cnt // 2
            for i in range(lo, lo + k):
                j = i + k
                if (up==1 and arr[i] > arr[j]) or (up == 0 and arr[i]<arr[j]):
                    arr[i], arr[j] = arr[j], arr[i]
                    yield arr
            for sub in bitonic_merge(lo, k, up):
                yield sub
            for sub in bitonic_merge(lo+k, k, up):
                yield sub
        
    if lo is None or cnt is None or up is None:
        lo, up = 0, 1
        n = cnt = len(arr)
        while n > 1:
            assert n % 2 == 0
            n //= 2
    if cnt > 1:
        k = cnt // 2
        for sub in bitonic_sort(arr, lo, k, 1):
            yield sub
        for sub in bitonic_sort(arr, lo + k, k, 0):
            yield sub
        for sub in bitonic_merge(lo, cnt, up):
            yield sub

def merge_sort(arr, lo=None, hi=None):
    """
    merge sort algorithm
    O(nlog(n))
    """
    if lo is None or hi is None:
        lo, hi = 0, len(arr)
    if hi - lo < 2:
        return
    
    mi = lo + (hi - lo) // 2
    for sub in merge_sort(arr, lo=lo, hi=mi):
        yield sub
    L = arr[lo:mi]
    for sub in merge_sort(arr, lo=mi, hi=hi):
        yield sub
    R = arr[mi:hi]
    i = j = 0
    k = lo
    
    while i < len(L) and j < len(R):
        if L[i] < R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1
        yield arr[:k] + L[i:] + R[j:] + arr[hi:]
    
    while i < len(L):
        arr[k] = L[i]
        i += 1
        k += 1
        yield arr[:k] + L[i:] + arr[hi:]
    
    while j < len(R):
        arr[k] = R[j]
        j += 1
        k += 1
        yield arr[:k] + R[j:] + arr[hi:]
    
def quick_sort(arr, lo=0, hi=None):
    """
    quick sort
    worst: O(n^2), average: O(n*log(n)), best: O(n*log(n))
    """
    def partition(arr, lo, hi):
        ret = list()
        i = lo
        x = arr[hi]
        for j in range(lo, hi):
            if arr[j] < x:
                arr[i], arr[j] = arr[j], arr[i]
                ret.append(arr[:])
                i += 1
        arr[i], arr[hi] = arr[hi], arr[i]
        ret.append(arr[:])
        return i, ret
    
    if hi is None:
        hi = len(arr) - 1
    if lo < hi:
        pi, ret = partition(arr, lo, hi)
        for sub in ret:
            yield sub
        for sub in quick_sort(arr, lo, pi - 1):
            yield sub
        for sub in quick_sort(arr, pi + 1, hi):
            yield sub
  
def heap_sort(arr): 
    """
    heap sort
    O(nlog(n)) for any scenario
    """
    def heapify(arr, n, i): 
        largest = i
        l, r = 2 * i + 1, 2 * i + 2
        if l < n and arr[i] < arr[l]: 
            largest = l 
        if r < n and arr[largest] < arr[r]: 
            largest = r 
        if largest != i: 
            arr[i], arr[largest] = arr[largest], arr[i]
            yield arr
            for sub in heapify(arr, n, largest):
                yield arr
    
    length = len(arr)
    for i in range(length, -1, -1): 
        for sub in heapify(arr, length, i):
            yield arr
    for i in range(length-1, 0, -1): 
        arr[i], arr[0] = arr[0], arr[i]
        yield arr
        for sub in heapify(arr, i, 0):
            yield arr

def tim_sort(arr, run=215):
    """
    tim sort
    O(nlog(n))
    """
    def merge(arr, lo, mi, hi):
        L = arr[lo:mi].copy()
        R = arr[mi:hi].copy()
        i = j = 0
        k = lo
        
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
            yield arr[:k] + L[i:] + R[j:] + arr[hi:]
        
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
            yield arr[:k] + L[i:] + arr[hi:]
        
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
            yield arr[:k] + R[j:] + arr[hi:]
    
    n = len(arr)
    
    for start in range(0, n, run):
        for sub in binsertion_sort(arr, start, min(start+run, n)):
            yield sub
    
    while run < n:
        for lo in range(0, n, 2 * run):
            mi = min(n, lo + run)
            hi = min(n, mi + run)
            for sub in merge(arr, lo, mi, hi):
                yield sub
        run *= 2

def pancake_sort(arr):
    """
    pancake sort
    O(n^2)
    """
    cur_size = len(arr)
    while cur_size:
        m = max(range(cur_size), key=lambda i: arr[i]) + 1
        if m != cur_size:
            arr[:m] = arr[:m][::-1]
            yield arr
            arr[:cur_size] = arr[:cur_size][::-1]
            yield arr
        cur_size -= 1

def gnome_sort(arr):
    """
    gnome sort - similar to implementation of insertion sort
    O(n^2)
    """
    i = 0
    while i < len(arr):
        if arr[i] >= arr[i - 1]:
            i += 1
        else:
            arr[i], arr[i - 1] = arr[i - 1], arr[i]
            i = 1 if i == 1 else i - 1
            yield arr
            
def stooge_sort(arr, lo=None, hi=None):
    """
    stooge sort
    O(n^2.709)
    """
    if lo is None or hi is None:
        lo, hi = 0, len(arr) - 1
    if lo >= hi:
        return
    if arr[lo] > arr[hi]:
        arr[lo], arr[hi] = arr[hi], arr[lo]
        yield arr
    if hi - lo > 1:
        t = (hi - lo + 1) // 3
        for sub in stooge_sort(arr, lo, (hi - t)):
            yield sub
        for sub in stooge_sort(arr, lo + t, (hi)):
            yield sub
        for sub in stooge_sort(arr, lo, (hi - t)):
            yield sub
    
def counting_sort(arr):
    """
    counting sort
    O(n + k) where k is the max digit
    """
    lo = min(arr)
    domain = max(arr) - lo + 1
    counts = [0] * domain
    for x in arr:
        counts[x - lo] += 1
    for i in range(1, domain):
        counts[i] += counts[i - 1]
    # keep original set of arr present at each iteration
    not_set = set(range(len(arr)))
    for x in arr[:]:
        counts[x - lo] -= 1
        i = counts[x - lo]
        not_set.remove(i)
        j = next((j for j, y in enumerate(arr) if y == x and j in not_set), i)
        arr[i], arr[j] = x, arr[i]
        yield arr
        
def lsd_sort(arr, base=10):
    """
    radix sort using least significant digit (lsd)
    O(d(n + k)) where b is base (10), d is log_b(largest), n is length of arr
    """
    lo = min(arr, default=0)
    hi = max(arr, default=0) - lo
    exp = 1
    while hi // exp:
        counts = [0] * base
        for x in arr:
            counts[(x - lo) // exp % base] += 1
        for i in range(1, base):
            counts[i] += counts[i - 1]
        # this isn't an exact radix sort, the set is to keep the original set
        # of numbers present in all bar graphs
        not_set = set(range(len(arr)))
        for x in arr[::-1]:
            index = (x - lo) // exp % base
            counts[index] -= 1
            i = counts[index]
            not_set.remove(i)
            j = next((j for j, y in enumerate(arr) if y==x and j in not_set),i)
            arr[i], arr[j] = x, arr[i]
            yield arr
        exp *= base

def msd_sort(arr, lo = None, hi = None, digit = None):
    """
    radix sort using most significant digit (msd)
    O(d(n + k)) where b is base (10), d is log_b(largest), n is length of arr
    """
    if lo is None or hi is None or digit is None:
        lo, hi = 0, len(arr)
        tmp = max(arr)
        digit = 1
        while tmp >= 10:
            tmp //= 10
            digit *= 10
    
    if digit == 0 or hi - lo < 2:
        return
    
    tmp = arr[lo:hi]
    counts = [0] * 10
    for x in tmp:
        counts[(x // digit) % 10] += 1
    for i in range(1, len(counts)):
        counts[i] += counts[i - 1]
    not_set = set(range(lo, hi))
    for x in tmp:
        index = (x // digit) % 10
        counts[index] -= 1
        i = counts[index] + lo
        not_set.remove(i)
        j = next((j for j, y in enumerate(arr) if y == x and j in not_set), i)
        arr[i], arr[j] = x, arr[i]
        yield arr
    
    i = lo
    pl = (arr[i] // digit) % 10
    for j in range(i + 1, hi):
        if (arr[j] // digit) % 10 != pl:
            for sub in msd_sort(arr, i, j, digit // 10):
                yield sub
            i = j
            pl = (arr[j] // digit) % 10
    for sub in msd_sort(arr, i, hi, digit // 10):
        yield sub

def bucket_sort(arr, buckets=10):
    """
    bucket sort
    worst: O(n^2), average: O(n + k), best: O(n + k)
    """
    lo = min(arr, default=0)
    hi = max(arr, default=0)
    sz = (hi - lo) // buckets + 1
    
    slots = [list() for _ in range(buckets)]
    for i, x in enumerate(arr, start=1):
        slots[(x - lo) // sz].append(x)
        yield [x for sub in slots for x in sub] + arr[i:]
    
    for i, x in enumerate([x for sub in slots for x in sub]):
        arr[i] = x
    
    i = 0
    for slot in slots:
        for j, x in enumerate(slot):
            if j and slot[j] < slot[j - 1]:
                k = bisect(slot[:j], slot[j])
                slot.insert(k, slot.pop(j))
                yield [x for sub in slots for x in sub]
        for y in slot:
            arr[i] = y
            i += 1


def cycle_sort(arr):
    """
    cycle sort
    O(n^2)
    modified to keep original set at each yield
    """
    original = Counter(arr)
    
    def overwrite():
        tmp = Counter(arr)
        missing = [[key]*val for key, val in (original - tmp).items()]
        missing = [item for sub in missing for item in sub]
        extra = [[key]*val for key, val in (tmp - original).items()]
        extra = [item for sub in extra for item in sub]
        tmp = arr[:]
        for i, x in zip(missing, extra):
            tmp[tmp.index(x)] = i
        return tmp
        
    for i, x in enumerate(arr):
        j = i
        for y in arr[i + 1:]:
            if y < x:
                j += 1
        if i == j: continue
    
        while x == arr[j]:
            j += 1
        arr[j], x = x, arr[j]
        yield overwrite()
        
        while j != i:
            j = i
            for y in arr[i + 1:]:
                if y < x:
                    j += 1
            while x == arr[j]:
                j += 1
            arr[j], x = x, arr[j]
            yield overwrite()

def circle_sort(arr):
    """
    circle sort
    worst: O(n log^2(n)), best: O(n log(n))
    """
    swaps = True
    
    def backend(lo, hi):
        nonlocal swaps
        if lo == hi:
            return
        
        i, j = lo, hi
        mi = (hi - lo) // 2
        while i < j:
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
                swaps = True
                yield arr
            i += 1
            j -= 1
        
        if i == j and arr[i] > arr[j + 1]:
            arr[i], arr[j + 1] = arr[j + 1], arr[i]
            swaps = True
            yield arr
        
        for update in backend(lo, lo + mi):
            yield update
        for update in backend(lo + mi + 1, hi):
            yield update
    
    while swaps:
        swaps = False
        for update in backend(0, len(arr) - 1):
            yield update

algorithms = [bubble_sort, cocktail_sort, comb_sort, bucket_sort, \
              insertion_sort, binsertion_sort, selection_sort, \
              shell_sort, odd_even_sort, spaghetti_sort, bitonic_sort, \
              merge_sort, quick_sort, heap_sort, tim_sort, pancake_sort, \
              gnome_sort, stooge_sort, counting_sort, lsd_sort, msd_sort, \
              perm_sort, bogo_sort, slow_sort, circle_sort, cycle_sort, \
              strand_sort, patience_sort, islow_sort, bogo_sort, perm_sort]

if __name__ == '__main__':
    """
    Driver code runs <PCOUNT> processes to create a mp4 with each algorithm
    on a copy of the <data> list. 
    import multiprocessing as mp
    import time
    from tqdm import tqdm
    
    PCOUNT = 4
    data = list(sample(range(1, 2049), 2048))
    
    pbar = tqdm(total=len(algorithms))
    
    with mp.Pool(PCOUNT) as p:
        results = [p.apply_async(animate_plot, \
            args=(algo, data[:])) for algo in algorithms]
        
        # update pbar
        while results:
            complete = [i for i, x in enumerate(results) if x.ready()]
            pbar.update(len(complete))
            for i in reversed(complete):
                results.pop(i)
            time.sleep(1)

    """