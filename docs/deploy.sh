#!/usr/bin/env bash
# Deploy the site to BOTH Vercel projects.
#
# The site lives at two URLs owned by two Vercel scopes:
#   visualmemory.vercel.app    project "visualmemory"       team vi-zuara
#   visualmemory.vizuara.ai    project "visualmemory-site"  scope sreedaths-projects
#
# Two projects exist because the vizuara.ai domain is registered under the
# personal scope (where its other subdomain sites live), and a project in the
# vi-zuara team cannot attach it. Always deploy with this script so the two
# URLs never drift apart.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .vercel

echo '{"projectId":"prj_KsXU3LLV72JmHZFSUJtzgpHf4q8O","orgId":"team_umoWxNTjn2Z7yMC7d1Vnt63W","projectName":"visualmemory"}' > .vercel/project.json
npx vercel deploy --prod --yes

echo '{"projectId":"prj_ptIjdyopxzBwEIcBJf5iGviLYeh6","orgId":"team_QFh9j5O2aEema26rqrrXhRuN","projectName":"visualmemory-site"}' > .vercel/project.json
npx vercel deploy --prod --yes

echo "deployed: https://visualmemory.vercel.app and https://visualmemory.vizuara.ai"
