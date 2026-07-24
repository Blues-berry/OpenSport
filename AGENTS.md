# Repository collaboration rules

- Use `master` as the default integration branch and `origin/master` as the
  default push target.
- Create or push another branch only when the user explicitly requests it.
- Do not push automatically. Commit locally first, report the commit, and push
  to `origin/master` only after the user explicitly authorizes that remote
  write.
- Never force-push `master`.
- Commit source code, tests, documentation, and small configuration files.
  Keep raw sensor data, generated features, models, logs, and runtime captures
  under the ignored `data/` and `imu_output/` trees.
