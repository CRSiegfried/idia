# Security

## Reporting a vulnerability

Please report a vulnerability privately, not in a public issue. Use GitHub's private
vulnerability reporting: open the repository's **Security** tab and choose **Report a
vulnerability**. Say what is affected, the idia version or commit, and how to
reproduce it.

idia is maintained by one person, so there is no fixed timeline. Expect an
acknowledgement within about a week; a fix, or an explanation of why none is needed,
follows as soon as it can be made and checked. Please keep the report private until a fix
is released or we agree it can be disclosed.

## Credentials and recorded cassettes

idia reads provider credentials from the arguments it is given or from the
environment, and never from a file. The test cassettes committed to this repository are
recordings of real wire exchanges, filtered of credentials when they are written: secret
headers are replaced and a secret the recording names, such as an account id in a URL, is
redacted; a recorded run checks each file it writes for the exact values of the credentials
it was given, and the test suite checks every committed cassette for the shapes of API keys
and bearer tokens.
If you find a credential in a cassette, report it as above.
