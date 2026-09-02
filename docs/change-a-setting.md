# Change a setting

A setting is any value a person might want to change before a run. The name of an experiment. A number of rounds. A threshold. You keep settings in the `config/` folder.

Here is the part that feels like magic. Every setting you write in `config/` shows up as a field on the run page. You write one line. A field appears. There is no extra wiring.

## Look at what is there

Open `config/experiment.yaml`. It looks like this.

```yaml
experiment:
  name: "my-experiment-v1"
  description: ""
```

Two settings. `name` and `description`. On the run page these become two fields called `experiment-name` and `experiment-description`. The rule for the field name is simple. Take the path, replace the dots with dashes.

![The experiment-name and experiment-description fields on the run form](img/form-experiment-fields.png)

## Add a new setting

Say you want people to choose a number of rounds. Add one line under `experiment`.

```yaml
experiment:
  name: "my-experiment-v1"
  description: ""
  rounds: 3
```

Save the file. That is the whole change. After you send it in, the run page will have a new field called `experiment-rounds` with a default of `3`.

Your step reads it the same way as any other setting.

```python
experiment = cfg["experiment"]
print(experiment["rounds"])
```

## Change a default

Want a different starting value. Just edit the number.

```yaml
  rounds: 5
```

Now the field shows up with `5` already filled in. People can still change it on the run page. You only changed the starting point.

## What can be a setting

Keep it to simple values. These work well.

- Text, like a name.
- Numbers, like `3` or `0.5`.
- True or false.

These do not work as fields.

- Long lists.
- Big blocks of nested data.

If you need one of those, ask your Novelcore contact. There is a way to do it with a dropdown of preset choices, and they can show you.

## A quick word on dropdowns

Sometimes you want a person to pick from a fixed set of choices rather than type free text. That is a dropdown. You make one by creating a small folder of choice files instead of a single value. It is a slightly more advanced move. When you need it, ask your Novelcore contact to walk you through it once. After that it is easy.

Next, send your change in. Go to [Send your change in](send-it-in.md).
