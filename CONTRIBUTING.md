# Contribution guidelines

Contributions to covfie are strongly welcomed and greatly appreciated from any interested party! If you would like to contribute, please read through this document; it's designed to help you get started and to make your first contribution as smooth as possible.

## Copyright

Any contributions to covfie are automatically licensed under the same license as the project itself. By contributing to covfie, you are agreeing to license your contribution under the project's chosen license.

## Attribution

It is important that contributors to covfie get proper attribution. To this end, please make sure that your commits are associated with your preferred name and that the e-mail address used in the author line of the commit is associated to your GitHub account. This way, GitHub will automatically attribute commits to you, and the commits will show up on your GitHub profile.

## Collaborating

The primary methods for communicating about the development of covfie are through the [CERN Mattermost channel](https://mattermost.web.cern.ch/acts/channels/covfie). If you don't have access to the CERN Mattermost server, you can also e-mail the authors or open a GitHub issue.

## Submitting patches

If you want to contribute to covfie, you can open a pull request into the official covfie repository. This will run a continuous integration (CI) job that ensures that the code compiles, works, and that the code is properly formatted. If you want to preempt any linting errors, you can use the pre-commit tool locally; installing pre-commit is straightforward and more information is available on [the pre-commit home page](https://pre-commit.com/).

Ideally, commits should follow the standard git best practices for commit messages, i.e. the title line should be 50 characters or fewer. A more elaborate description follows two newlines and is wrapped at 72 characters. Commit titles should be in the imperative mood, while the description can be written following author preference.

Generally, merge commits are preferred over rebasing or squashing, but this requires every single commit in the pull request to pass the CI. If this is not the case, no problem! We can still squash your pull request. However, squashing pull requests via the GitHub interface prevents the commit that you have written and potentially signed from ending up directly in the repository; instead, a commit committed by GitHub will be created. Merging of any pull request is predicated on approval from a core developer.

When creating pull requests, less is more. If your pull request addresses two separate issues, please split the commit in two. The mental load of reviewing two small pull requests is much lower than reviewing one big pull request.
