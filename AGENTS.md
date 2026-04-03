## Code Guidelines

- Avoid try/except blocks unless it's really necessary.  It's fine that a program fails if something goes wrong as this helps us to catch non-obvious bugs and unforeseen side-effects earlier. You can add try catch on code that explicitly aims to be fault tolerant like adding retry mechanisms or explicit and intentional robustness. 

- Do not add unnecessary comments. Especially do not try to explain code change that reflect your work process, do not refer to old code. "The code used to do that but now we are doing this" is not a pattern we want. Instead prefer to use targeted comments sparingly to explain ambiguous code.


## Zen of Python
remember the zen of python when writing code.

```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

## Running code

- All code should be runnable with `uv run` or `uv run <command>`.
- All dependencies should already be installed and pin in the lock file. If not, add it to pyproject.toml and run `uv sync --all-extras` to install it.