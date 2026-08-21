# 🦥 Contributing to Unsloth

Thank you for not only using Unsloth but also for being interested in helping out! We value all contributions, whether they come in the form of code, ideas, support for others or just by simply spreading the word of Unsloth! 💕

- **[Support the Community](https://github.com/unslothai/unsloth/issues)**: Answer questions, review pull requests, or assist others in discussions.
- **Fix Bugs**: Identify and resolve issues with the existing codebase.
- **Submit Ideas**: Request new features or share enhancements you'd like to see.
- **Develop Features**: Implement new functionality or improve existing tools which can be done via PRs.
- **[Improve Documentation](https://docs.unsloth.ai/)**: Help by creating guides, FAQs, or enhancing clarity.

One of the best ways to support us is by spreading the word about Unsloth! Share how it’s powering your amazing projects in blog posts or social media, and inspire others to explore its potential. Even a simple star on our repo goes a long way in showing your support and helping the community grow. 🌟

## Submitting Issues
If you find a bug or have a feature idea, we’d love to hear from you! Here’s how to make your submission stand out:

### Reporting Bugs
1. **Search First**: Check if the issue has already been reported using GitHub’s search bar under Issues.
2. **Details Matter**: Is this on Google Colab, Kaggle, or on another platform service? Are you using Unsloth's official notebook? Include your OS, Python version, and other relevant details. For bugs, a concise code snippet that reproduces the issue is incredibly helpful.
3. **Be Thorough**: Attach screenshots, traceback logs, or any additional information that might speed up resolution.

## Spread the Word
Your support extends beyond code:
- Spread the word by writing about Unsloth in blogs or social media.
- Share how Unsloth powers your projects.
- Star our repository to show your appreciation.

Finally, please be mindful of our [Code of Conduct](https://github.com/unslothai/unsloth/blob/main/CODE_OF_CONDUCT.md) to ensure a welcoming and inclusive environment for everyone.

Thank you so much for reading and we hope you have lots of fun using Unsloth! 🦥


## Pull Request Guidelines
- Keep PRs focused on a single change
- Include a concise description and motivation
- Link related issues when applicable

### Changes to Studio's UI

A pull request that touches the Studio frontend must leave the interface behaving the same
way it did before. Reviewers should be able to assume that a change described as a
performance fix is a performance fix and nothing else.

There are two exemptions.

1. **A dramatic performance improvement can justify a deliberate UI difference.** State the
   difference plainly in the PR body, say what it costs the user, and attach the measurement
   that justifies it. A difference that is not called out is a regression, however fast the
   PR is.
2. **A difference that exists only off screen is fine.** Rendering only what is visible is an
   accepted technique, not a parity violation. Windowing, deferring off-screen work and
   unmounting rows the user cannot see are all allowed, provided everything inside the
   viewport is identical to what it was before.

The second exemption is narrower than it sounds, and the boundary is where these changes go
wrong in practice:

- An element that is partly on screen is visible, and so is one that scrolls into view during
  an interaction.
- Selection, clipboard, native find-in-page and printing are whole-document operations. A
  user pressing Ctrl+A, Ctrl+C or Ctrl+F is asking about the conversation, not about the
  viewport, so off-screen content still has to participate. Truncating a copied thread is a
  data-loss bug, not an off-screen rendering difference.
- Scroll geometry is visible. A scrollbar that no longer describes the length of the thread
  is a UI change even though the thing that shrank was off screen.

If you are changing what the thread renders, measure at the 100K rung or larger against a
concurrent control, and say which of the two exemptions you are relying on, or that you are
relying on neither.
