# Remove a step

Taking a step out is quick. It is the reverse of adding one. Do these in order so nothing is left dangling.

We will remove the `say-goodbye` step we added earlier. Swap in your own step name.

## Step 1. Remove the line from pipeline.py

Open `pipeline.py`. Delete the line for the step.

```python
with pipeline("ml-pipeline") as p:
    hello = step("hello-world", reads=["experiment"])
    goodbye = step("say-goodbye", reads=["experiment"], needs=[hello])   # delete this line
```

## Step 2. Check if any other step needed it

If another step had `needs=[goodbye]`, remove `goodbye` from that list. Otherwise a step will be waiting for something that no longer exists.

In our small example nothing else needed it, so there is nothing to do here. In a bigger pipeline, scan the file for the step name and clean it out of every `needs`.

## Step 3. Delete the step folder

Delete the whole folder for the step.

```bash
rm -rf steps/say_goodbye
```

You can also right click the folder in Visual Studio Code and choose Delete.

## Step 4. Clean up its settings (only if it had any)

If your step had its own settings file in `config/`, and no other step uses those settings, remove them too.

- Delete the settings file, for example `config/goodbye.yaml`.
- Open `config/config.yaml` and remove the line that listed it under `defaults`.

The `say-goodbye` step read the shared `experiment` settings, so there is nothing to delete here. Skip this step.

## Step 5. That is it

Send the change in the same way as always. See [Send your change in](send-it-in.md).

!!! warning "If you forget a cleanup"
    Say you delete a step but leave another step still pointing at it, or still reading a settings section you removed. The check on your request will fail and name the exact step and the exact setting. Nothing breaks quietly. Read the message, fix the one thing it names, and push again.
