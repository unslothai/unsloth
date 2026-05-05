# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GraphQL queries for GitHub data scraping.

GitHub's GraphQL rejects queries that define unused fragments, so each query
only includes the fragments it actually references.
"""

# ---- Fragments (kept as raw strings, composed per query) ----
F_ACTOR = """
fragment ActorFields on Actor {
  __typename
  login
  url
  avatarUrl
  ... on User { id databaseId name }
  ... on Bot { id databaseId }
  ... on Organization { id databaseId name }
}
"""

F_LABEL = """
fragment LabelFields on Label {
  id
  name
  color
  description
  createdAt
}
"""

F_TIMELINE = """
fragment TimelineItem on IssueTimelineItems {
  __typename
  ... on Node { id }
  ... on AddedToProjectEvent { createdAt actor { ...ActorFields } }
  ... on AssignedEvent { createdAt actor { ...ActorFields } assignee { __typename ... on User { login } ... on Bot { login } } }
  ... on ClosedEvent { createdAt actor { ...ActorFields } stateReason closer { __typename ... on Commit { oid url } ... on PullRequest { number url } } }
  ... on CommentDeletedEvent { createdAt actor { ...ActorFields } }
  ... on ConnectedEvent { createdAt actor { ...ActorFields } source { __typename ... on Issue { number url repository { nameWithOwner } } ... on PullRequest { number url repository { nameWithOwner } } } subject { __typename ... on Issue { number url } ... on PullRequest { number url } } }
  ... on ConvertedNoteToIssueEvent { createdAt actor { ...ActorFields } }
  ... on CrossReferencedEvent { createdAt actor { ...ActorFields } isCrossRepository willCloseTarget source { __typename ... on Issue { number url repository { nameWithOwner } title } ... on PullRequest { number url repository { nameWithOwner } title } } }
  ... on DemilestonedEvent { createdAt actor { ...ActorFields } milestoneTitle }
  ... on DisconnectedEvent { createdAt actor { ...ActorFields } subject { __typename ... on Issue { number url } ... on PullRequest { number url } } source { __typename ... on Issue { number url } ... on PullRequest { number url } } }
  ... on IssueComment { id databaseId createdAt updatedAt author { ...ActorFields } body url reactionGroups { content reactors { totalCount } } }
  ... on LabeledEvent { createdAt actor { ...ActorFields } label { name color } }
  ... on LockedEvent { createdAt actor { ...ActorFields } lockReason }
  ... on MarkedAsDuplicateEvent { createdAt actor { ...ActorFields } canonical { __typename ... on Issue { number url } ... on PullRequest { number url } } }
  ... on MentionedEvent { createdAt actor { ...ActorFields } }
  ... on MilestonedEvent { createdAt actor { ...ActorFields } milestoneTitle }
  ... on MovedColumnsInProjectEvent { createdAt actor { ...ActorFields } }
  ... on PinnedEvent { createdAt actor { ...ActorFields } }
  ... on ReferencedEvent { createdAt actor { ...ActorFields } commit { oid url } commitRepository { nameWithOwner } }
  ... on RemovedFromProjectEvent { createdAt actor { ...ActorFields } }
  ... on RenamedTitleEvent { createdAt actor { ...ActorFields } previousTitle currentTitle }
  ... on ReopenedEvent { createdAt actor { ...ActorFields } }
  ... on SubscribedEvent { createdAt actor { ...ActorFields } }
  ... on TransferredEvent { createdAt actor { ...ActorFields } fromRepository { nameWithOwner } }
  ... on UnassignedEvent { createdAt actor { ...ActorFields } assignee { __typename ... on User { login } ... on Bot { login } } }
  ... on UnlabeledEvent { createdAt actor { ...ActorFields } label { name color } }
  ... on UnlockedEvent { createdAt actor { ...ActorFields } }
  ... on UnmarkedAsDuplicateEvent { createdAt actor { ...ActorFields } }
  ... on UnpinnedEvent { createdAt actor { ...ActorFields } }
  ... on UnsubscribedEvent { createdAt actor { ...ActorFields } }
  ... on UserBlockedEvent { createdAt actor { ...ActorFields } blockDuration }
}
"""

F_PR_TIMELINE = """
fragment PRTimelineItem on PullRequestTimelineItems {
  __typename
  ... on Node { id }
  ... on AssignedEvent { createdAt actor { ...ActorFields } assignee { __typename ... on User { login } ... on Bot { login } } }
  ... on AutoMergeDisabledEvent { createdAt actor { ...ActorFields } reason }
  ... on AutoMergeEnabledEvent { createdAt actor { ...ActorFields } }
  ... on AutoRebaseEnabledEvent { createdAt actor { ...ActorFields } }
  ... on AutoSquashEnabledEvent { createdAt actor { ...ActorFields } }
  ... on AutomaticBaseChangeFailedEvent { createdAt actor { ...ActorFields } oldBase newBase }
  ... on AutomaticBaseChangeSucceededEvent { createdAt actor { ...ActorFields } oldBase newBase }
  ... on BaseRefChangedEvent { createdAt actor { ...ActorFields } previousRefName currentRefName }
  ... on BaseRefDeletedEvent { createdAt actor { ...ActorFields } baseRefName }
  ... on BaseRefForcePushedEvent { createdAt actor { ...ActorFields } beforeCommit { oid } afterCommit { oid } ref { name } }
  ... on ClosedEvent { createdAt actor { ...ActorFields } stateReason }
  ... on CommentDeletedEvent { createdAt actor { ...ActorFields } }
  ... on ConnectedEvent { createdAt actor { ...ActorFields } source { __typename ... on Issue { number url } ... on PullRequest { number url } } subject { __typename ... on Issue { number url } ... on PullRequest { number url } } }
  ... on ConvertToDraftEvent { createdAt actor { ...ActorFields } }
  ... on CrossReferencedEvent { createdAt actor { ...ActorFields } isCrossRepository willCloseTarget source { __typename ... on Issue { number url repository { nameWithOwner } title } ... on PullRequest { number url repository { nameWithOwner } title } } }
  ... on DemilestonedEvent { createdAt actor { ...ActorFields } milestoneTitle }
  ... on DeployedEvent { createdAt actor { ...ActorFields } }
  ... on DeploymentEnvironmentChangedEvent { createdAt actor { ...ActorFields } }
  ... on DisconnectedEvent { createdAt actor { ...ActorFields } subject { __typename ... on Issue { number url } ... on PullRequest { number url } } source { __typename ... on Issue { number url } ... on PullRequest { number url } } }
  ... on HeadRefDeletedEvent { createdAt actor { ...ActorFields } headRefName }
  ... on HeadRefForcePushedEvent { createdAt actor { ...ActorFields } beforeCommit { oid } afterCommit { oid } ref { name } }
  ... on HeadRefRestoredEvent { createdAt actor { ...ActorFields } }
  ... on IssueComment { id databaseId createdAt updatedAt author { ...ActorFields } body url reactionGroups { content reactors { totalCount } } }
  ... on LabeledEvent { createdAt actor { ...ActorFields } label { name color } }
  ... on LockedEvent { createdAt actor { ...ActorFields } lockReason }
  ... on MarkedAsDuplicateEvent { createdAt actor { ...ActorFields } canonical { __typename ... on Issue { number url } ... on PullRequest { number url } } }
  ... on MentionedEvent { createdAt actor { ...ActorFields } }
  ... on MergedEvent { createdAt actor { ...ActorFields } commit { oid url } mergeRefName }
  ... on MilestonedEvent { createdAt actor { ...ActorFields } milestoneTitle }
  ... on MovedColumnsInProjectEvent { createdAt actor { ...ActorFields } }
  ... on PinnedEvent { createdAt actor { ...ActorFields } }
  ... on PullRequestCommit { commit { oid url message author { user { login } date } committedDate } }
  ... on PullRequestCommitCommentThread { commit { oid } }
  ... on PullRequestReview { id databaseId createdAt submittedAt author { ...ActorFields } body state url reactionGroups { content reactors { totalCount } } }
  ... on PullRequestReviewThread { id isResolved isOutdated path line diffSide }
  ... on PullRequestRevisionMarker { createdAt lastSeenCommit { oid } }
  ... on ReadyForReviewEvent { createdAt actor { ...ActorFields } }
  ... on ReferencedEvent { createdAt actor { ...ActorFields } commit { oid url } commitRepository { nameWithOwner } }
  ... on RenamedTitleEvent { createdAt actor { ...ActorFields } previousTitle currentTitle }
  ... on ReopenedEvent { createdAt actor { ...ActorFields } }
  ... on ReviewDismissedEvent { createdAt actor { ...ActorFields } dismissalMessage previousReviewState }
  ... on ReviewRequestRemovedEvent { createdAt actor { ...ActorFields } requestedReviewer { __typename ... on User { login } ... on Team { name } } }
  ... on ReviewRequestedEvent { createdAt actor { ...ActorFields } requestedReviewer { __typename ... on User { login } ... on Team { name } } }
  ... on SubscribedEvent { createdAt actor { ...ActorFields } }
  ... on TransferredEvent { createdAt actor { ...ActorFields } fromRepository { nameWithOwner } }
  ... on UnassignedEvent { createdAt actor { ...ActorFields } assignee { __typename ... on User { login } ... on Bot { login } } }
  ... on UnlabeledEvent { createdAt actor { ...ActorFields } label { name color } }
  ... on UnlockedEvent { createdAt actor { ...ActorFields } }
  ... on UnmarkedAsDuplicateEvent { createdAt actor { ...ActorFields } }
  ... on UnpinnedEvent { createdAt actor { ...ActorFields } }
  ... on UnsubscribedEvent { createdAt actor { ...ActorFields } }
  ... on UserBlockedEvent { createdAt actor { ...ActorFields } blockDuration }
}
"""


def _q(parts: list[str], body: str) -> str:
    return "\n".join(parts + [body])


ISSUES_PAGE_QUERY = _q(
    [F_ACTOR, F_LABEL, F_TIMELINE],
    """
query IssuesPage($owner: String!, $name: String!, $first: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    issues(first: $first, after: $after, orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      totalCount
      nodes {
        id databaseId number title body state stateReason
        createdAt updatedAt closedAt
        url
        author { ...ActorFields }
        editor { ...ActorFields }
        labels(first: 50) { nodes { ...LabelFields } }
        assignees(first: 20) { nodes { login id } }
        milestone { title number state dueOn }
        reactionGroups { content reactors { totalCount } }
        comments(first: 100) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes {
            id databaseId createdAt updatedAt url body
            author { ...ActorFields }
            editor { ...ActorFields }
            reactionGroups { content reactors { totalCount } }
          }
        }
        timelineItems(first: 100) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes { ...TimelineItem }
        }
        trackedInIssues(first: 20) { totalCount nodes { number url repository { nameWithOwner } } }
        trackedIssues(first: 20) { totalCount nodes { number url repository { nameWithOwner } } }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
""",
)

PRS_PAGE_QUERY = _q(
    [F_ACTOR, F_LABEL, F_PR_TIMELINE],
    """
query PRsPage($owner: String!, $name: String!, $first: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: $first, after: $after, orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      totalCount
      nodes {
        id databaseId number title body state isDraft
        createdAt updatedAt closedAt mergedAt
        url
        headRefName headRefOid
        baseRefName baseRefOid
        additions deletions changedFiles
        mergeable merged mergeStateStatus
        author { ...ActorFields }
        editor { ...ActorFields }
        mergedBy { ...ActorFields }
        labels(first: 50) { nodes { ...LabelFields } }
        assignees(first: 20) { nodes { login id } }
        milestone { title number state dueOn }
        reactionGroups { content reactors { totalCount } }
        closingIssuesReferences(first: 20) { totalCount nodes { number url repository { nameWithOwner } title } }
        comments(first: 100) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes {
            id databaseId createdAt updatedAt url body
            author { ...ActorFields }
            editor { ...ActorFields }
            reactionGroups { content reactors { totalCount } }
          }
        }
        reviewThreads(first: 50) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes {
            id isResolved isOutdated path line diffSide
            comments(first: 50) {
              totalCount
              pageInfo { hasNextPage endCursor }
              nodes {
                id databaseId createdAt updatedAt url body path diffHunk
                author { ...ActorFields }
                editor { ...ActorFields }
                position originalPosition line originalLine
                commit { oid }
                reactionGroups { content reactors { totalCount } }
              }
            }
          }
        }
        reviews(first: 50) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes {
            id databaseId state createdAt submittedAt body url
            author { ...ActorFields }
            reactionGroups { content reactors { totalCount } }
          }
        }
        commits(first: 100) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes {
            commit {
              oid
              message
              messageHeadline
              committedDate
              authoredDate
              author { name email user { login } date }
              committer { name email user { login } date }
              additions deletions changedFilesIfAvailable
              parents(first: 3) { nodes { oid } }
            }
          }
        }
        files(first: 100) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes {
            path additions deletions changeType
          }
        }
        timelineItems(first: 100) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes { ...PRTimelineItem }
        }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
""",
)

PRS_PAGE_QUERY_LIGHT = _q(
    [F_ACTOR, F_LABEL],
    """
query PRsPageLight($owner: String!, $name: String!, $first: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: $first, after: $after, orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      totalCount
      nodes {
        id databaseId number title body state isDraft
        createdAt updatedAt closedAt mergedAt
        url
        author { ...ActorFields }
        labels(first: 50) { nodes { ...LabelFields } }
        comments(first: 30) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes {
            id databaseId createdAt updatedAt url body
            author { ...ActorFields }
          }
        }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
""",
)

ISSUES_PAGE_QUERY_LIGHT = _q(
    [F_ACTOR, F_LABEL],
    """
query IssuesPageLight($owner: String!, $name: String!, $first: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    issues(first: $first, after: $after, orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      totalCount
      nodes {
        id databaseId number title body state
        createdAt updatedAt closedAt
        url
        author { ...ActorFields }
        labels(first: 50) { nodes { ...LabelFields } }
        comments(first: 30) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes {
            id databaseId createdAt updatedAt url body
            author { ...ActorFields }
          }
        }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
""",
)

ISSUE_COMMENTS_QUERY = _q(
    [F_ACTOR],
    """
query IssueComments($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    issueOrPullRequest(number: $number) {
      __typename
      ... on Issue {
        comments(first: 100, after: $after) {
          pageInfo { hasNextPage endCursor }
          nodes {
            id databaseId createdAt updatedAt url body
            author { ...ActorFields }
            editor { ...ActorFields }
            reactionGroups { content reactors { totalCount } }
          }
        }
      }
      ... on PullRequest {
        comments(first: 100, after: $after) {
          pageInfo { hasNextPage endCursor }
          nodes {
            id databaseId createdAt updatedAt url body
            author { ...ActorFields }
            editor { ...ActorFields }
            reactionGroups { content reactors { totalCount } }
          }
        }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
""",
)

ISSUE_TIMELINE_QUERY = _q(
    [F_ACTOR, F_TIMELINE],
    """
query IssueTimeline($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      timelineItems(first: 100, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes { ...TimelineItem }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
""",
)

PR_TIMELINE_QUERY = _q(
    [F_ACTOR, F_PR_TIMELINE],
    """
query PRTimeline($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      timelineItems(first: 100, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes { ...PRTimelineItem }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
""",
)

PR_COMMITS_QUERY = """
query PRCommits($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      commits(first: 100, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          commit {
            oid message messageHeadline committedDate authoredDate
            author { name email user { login } date }
            committer { name email user { login } date }
            additions deletions changedFilesIfAvailable
            parents(first: 3) { nodes { oid } }
          }
        }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
"""

PR_FILES_QUERY = """
query PRFiles($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      files(first: 100, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes { path additions deletions changeType }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
"""

PR_REVIEW_THREADS_QUERY = _q(
    [F_ACTOR],
    """
query PRReviewThreads($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 50, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id isResolved isOutdated path line diffSide
          comments(first: 50) {
            totalCount
            nodes {
              id databaseId createdAt updatedAt url body path diffHunk
              author { ...ActorFields }
              editor { ...ActorFields }
              position originalPosition line originalLine
              commit { oid }
              reactionGroups { content reactors { totalCount } }
            }
          }
        }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
""",
)

DISCUSSIONS_PAGE_QUERY = _q(
    [F_ACTOR, F_LABEL],
    """
query DiscussionsPage($owner: String!, $name: String!, $first: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: $first, after: $after, orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      totalCount
      nodes {
        id databaseId number title body
        createdAt updatedAt url
        author { ...ActorFields }
        editor { ...ActorFields }
        locked
        answerChosenAt
        closed closedAt
        category { id name emoji description isAnswerable }
        labels(first: 30) { nodes { ...LabelFields } }
        upvoteCount
        answer { id databaseId body author { ...ActorFields } createdAt url }
        reactionGroups { content reactors { totalCount } }
        comments(first: 50) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes {
            id databaseId body createdAt updatedAt url
            author { ...ActorFields }
            editor { ...ActorFields }
            upvoteCount
            isAnswer
            reactionGroups { content reactors { totalCount } }
            replies(first: 50) {
              totalCount
              pageInfo { hasNextPage endCursor }
              nodes {
                id databaseId body createdAt updatedAt url
                author { ...ActorFields }
                editor { ...ActorFields }
                reactionGroups { content reactors { totalCount } }
              }
            }
          }
        }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
""",
)

DISCUSSION_COMMENTS_QUERY = _q(
    [F_ACTOR],
    """
query DiscussionComments($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    discussion(number: $number) {
      comments(first: 50, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id databaseId body createdAt updatedAt url
          author { ...ActorFields }
          editor { ...ActorFields }
          upvoteCount
          isAnswer
          reactionGroups { content reactors { totalCount } }
          replies(first: 50) {
            totalCount
            nodes {
              id databaseId body createdAt updatedAt url
              author { ...ActorFields }
              editor { ...ActorFields }
              reactionGroups { content reactors { totalCount } }
            }
          }
        }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
""",
)

DISCUSSION_REPLIES_QUERY = _q(
    [F_ACTOR],
    """
query DiscussionReplies($commentId: ID!, $after: String) {
  node(id: $commentId) {
    ... on DiscussionComment {
      replies(first: 50, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id databaseId body createdAt updatedAt url
          author { ...ActorFields }
          editor { ...ActorFields }
          reactionGroups { content reactors { totalCount } }
        }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
""",
)

COMMITS_PAGE_QUERY = """
query CommitsPage($owner: String!, $name: String!, $first: Int!, $after: String, $branch: String!) {
  repository(owner: $owner, name: $name) {
    ref(qualifiedName: $branch) {
      target {
        ... on Commit {
          history(first: $first, after: $after) {
            pageInfo { hasNextPage endCursor }
            totalCount
            nodes {
              oid
              message
              messageHeadline
              committedDate
              authoredDate
              url
              additions deletions changedFilesIfAvailable
              author { name email date user { login id } }
              committer { name email date user { login id } }
              parents(first: 3) { nodes { oid } }
              associatedPullRequests(first: 5) { nodes { number url state } }
            }
          }
        }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
"""

RELEASES_QUERY = _q(
    [F_ACTOR],
    """
query Releases($owner: String!, $name: String!, $first: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    releases(first: $first, after: $after, orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        id databaseId name tagName description
        createdAt publishedAt updatedAt
        isDraft isPrerelease isLatest
        url
        author { ...ActorFields }
        tagCommit { oid url }
        reactionGroups { content reactors { totalCount } }
        releaseAssets(first: 50) {
          nodes { name contentType size downloadUrl createdAt updatedAt }
        }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
""",
)

LABELS_QUERY = _q(
    [F_LABEL],
    """
query LabelsList($owner: String!, $name: String!, $first: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    labels(first: $first, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes { ...LabelFields }
    }
  }
  rateLimit { cost remaining resetAt }
}
""",
)

MILESTONES_QUERY = """
query Milestones($owner: String!, $name: String!, $first: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    milestones(first: $first, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes {
        id number title description state
        createdAt updatedAt closedAt dueOn
        creator { login }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
"""

REPO_META_QUERY = """
query RepoMeta($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    id databaseId name nameWithOwner description url
    createdAt updatedAt pushedAt
    isArchived isDisabled isFork isPrivate
    primaryLanguage { name }
    languages(first: 20, orderBy: {field: SIZE, direction: DESC}) {
      edges { size node { name } }
      totalSize
    }
    stargazerCount forkCount watchers { totalCount }
    diskUsage
    licenseInfo { key name }
    homepageUrl
    defaultBranchRef { name }
  }
  rateLimit { cost remaining resetAt }
}
"""
