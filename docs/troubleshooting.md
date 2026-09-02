# When something breaks

Things go wrong. That is normal. The platform is built to tell you exactly what happened in plain words. This page lists the problems people hit most, and the fix for each.

The golden habit. When a check fails, click **Details** and read the message. It almost always names the one thing to fix.

## My new setting did not show up on the run page

Only simple values become fields. Text, numbers, true or false. If you tucked your setting inside a long list or a deep block of data, it will not appear. Move it to a plain value, or ask your contact about using a dropdown instead.

## The check on my request failed

Open the check with **Details** and read the reason. Here are the usual causes.

**A step is missing its Dockerfile.** You added the step to `pipeline.py` but there is no `Dockerfile` in its folder. Add the Dockerfile. See [Add a step](add-a-step.md).

**A step reads a setting that does not exist.** Maybe you renamed a settings section but a step still lists the old name in its `reads`. Update the step so it reads the correct name.

**A step needs another step that is gone.** You deleted a step but another one still points at it with `needs`. Remove it from that list.

In every case the message names the exact step and the exact setting. Fix that one thing, push again, and the check runs itself.

## I renamed a setting and now it fails

That is a safety net doing its job. A step is still reading the old name. The message tells you which step. Open that step and update its `reads` line, and the `READS` line in its code, to the new name.

## My step did not rebuild after I pushed

The platform only rebuilds a step when the code in its folder changed. If you only edited settings or `pipeline.py`, no rebuild is needed. Your change still went live. This is expected, not a bug.

## My run failed with a message about an invalid value

Some fields accept only a fixed set of choices. If you type something outside that set, the run stops right away and lists the allowed values. Pick one of the listed values. This check exists so a typo does not waste a whole run.

## A step turned red during a run

Click the red box in the run view. It opens the log for that step. The log shows what the step printed and where it stopped. Read the last few lines first. That is usually where the problem is.

## I am stuck and the message does not help

That happens. Copy the message, note what you were doing, and send it to your Novelcore contact. A clear description and the exact message is all they need to help fast.

## A short checklist before you push

Run through this and you will avoid most failures.

- [ ] My step name uses dashes, my folder uses underscores.
- [ ] My step has a `Dockerfile`.
- [ ] The `reads` in `pipeline.py` matches the `READS` in my code.
- [ ] Every `needs` points at a step that still exists.
- [ ] I saved every file before running `git add`.
