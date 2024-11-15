\ndef ratios(s):\n    '''Yield the ratios between adjacent values from iterator s.'''\n    prev = next(s)\n    for current in s:\n        yield prev / current\n        prev = current\n```"
