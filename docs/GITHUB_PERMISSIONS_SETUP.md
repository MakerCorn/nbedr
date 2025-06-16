# GitHub Actions & Package Permissions Setup

## Current Issue
If you're still seeing "installation not allowed to Create organization package" after the workflow permissions were added, you need to configure repository-level permissions.

## Step-by-Step Setup

### 1. Repository Actions Permissions

Visit: `https://github.com/MakerCorn/nbedr/settings/actions`

#### Actions Permissions
- ✅ Select **"Allow all actions and reusable workflows"**
- OR select **"Allow local actions..."** with appropriate GitHub Actions selected

#### Workflow Permissions
Choose **ONE** of these options:

**Option A: Full Permissions (Recommended)**
- ✅ Select **"Read and write permissions"**
- ✅ Check **"Allow GitHub Actions to create and approve pull requests"**

**Option B: Minimal Permissions**
- ✅ Select **"Read repository contents and packages permissions"**
- ✅ Check **"Allow GitHub Actions to create and approve pull requests"**

### 2. Organization Package Permissions (If Applicable)

If your repository is part of an organization, visit:
`https://github.com/orgs/MakerCorn/settings/packages`

#### Package Creation
- ✅ Allow members to create public packages
- ✅ Allow members to create private packages (if needed)

#### Package Access
- ✅ Set appropriate visibility defaults
- ✅ Configure package deletion policies

### 3. Personal Access Token (Alternative Method)

If repository settings don't resolve the issue, you can use a Personal Access Token:

#### Create PAT
1. Go to `https://github.com/settings/tokens`
2. Click **"Generate new token (classic)"**
3. Select these scopes:
   - ✅ `write:packages`
   - ✅ `read:packages`
   - ✅ `repo` (if private repository)

#### Add to Repository Secrets
1. Go to `https://github.com/MakerCorn/nbedr/settings/secrets/actions`
2. Click **"New repository secret"**
3. Name: `GHCR_TOKEN`
4. Value: Your PAT token

#### Update Workflow
Replace the login step in `.github/workflows/ci.yml`:
```yaml
- name: Login to GitHub Container Registry
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  uses: docker/login-action@v3
  with:
    registry: ghcr.io
    username: ${{ github.actor }}
    password: ${{ secrets.GHCR_TOKEN }}  # Changed from GITHUB_TOKEN
```

### 4. Troubleshooting

#### Check Current Permissions
Add this temporary step to your workflow to debug:
```yaml
- name: Check permissions
  run: |
    echo "GITHUB_TOKEN permissions:"
    curl -H "Authorization: token ${{ secrets.GITHUB_TOKEN }}" \
         -H "Accept: application/vnd.github.v3+json" \
         https://api.github.com/repos/${{ github.repository }}
```

#### Verify Package Visibility
1. After a successful push, check: `https://github.com/MakerCorn/nbedr/pkgs/container/nbedr`
2. Ensure package visibility is set correctly

#### Common Issues
- **Organization blocks package creation**: Check org settings
- **Repository is private but package is public**: Adjust visibility settings
- **GITHUB_TOKEN lacks permissions**: Use PAT method above
- **Branch protection rules**: May interfere with token permissions

### 5. Verification

After making changes:
1. Trigger a new workflow run by pushing to main
2. Check the build logs for successful authentication
3. Verify the package appears in GitHub Container Registry
4. Test pulling the image: `docker pull ghcr.io/makercorn/nbedr:latest`

## Quick Fix Commands

If you prefer command line configuration:

```bash
# Enable Actions (requires GitHub CLI)
gh api repos/MakerCorn/nbedr/actions/permissions \
  --method PUT \
  --field enabled=true \
  --field allowed_actions=all

# Set workflow permissions
gh api repos/MakerCorn/nbedr/actions/permissions/workflow \
  --method PUT \
  --field default_workflow_permissions=write \
  --field can_approve_pull_request_reviews=true
```