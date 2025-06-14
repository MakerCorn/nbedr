"""
SharePoint Online input source implementation.
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

try:
    import requests
    from msal import ConfidentialClientApplication, PublicClientApplication

    REQUESTS_AVAILABLE = True
    MSAL_AVAILABLE = True
except ImportError as e:
    missing = str(e).split("'")[1]
    if "requests" in missing:
        REQUESTS_AVAILABLE = False
    if "msal" in missing:
        MSAL_AVAILABLE = False

from .base import BaseInputSource, SourceDocument, SourceValidationError


class SharePointInputSource(BaseInputSource):
    """Input source for SharePoint Online document libraries."""

    def __init__(self, config):
        """Initialize SharePoint input source."""
        super().__init__(config)

        if not REQUESTS_AVAILABLE:
            raise SourceValidationError(
                "requests is required for SharePoint input sources. Install with: pip install requests"
            )

        if not MSAL_AVAILABLE:
            raise SourceValidationError("msal is required for SharePoint input sources. Install with: pip install msal")

        # Parse SharePoint URL
        self.site_url, self.library_path = self._parse_sharepoint_uri(config.source_uri)

        # Authentication components
        self.access_token = None
        self.token_expires_at = None
        self.session = requests.Session()

        # SharePoint API endpoints
        self.tenant_url = self._extract_tenant_url(self.site_url)
        self.site_id = None
        self.library_id = None

    def _parse_sharepoint_uri(self, uri: str) -> tuple[str, str]:
        """Parse SharePoint URI into site URL and library path."""
        # Expected format: https://tenant.sharepoint.com/sites/sitename/Shared Documents/folder
        # or: https://tenant.sharepoint.com/sites/sitename/Documents

        parsed = urlparse(uri)
        if not parsed.scheme or not parsed.netloc:
            raise SourceValidationError(f"Invalid SharePoint URL: {uri}")

        if "sharepoint.com" not in parsed.netloc:
            raise SourceValidationError(f"URL does not appear to be a SharePoint site: {uri}")

        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) < 2 or path_parts[0] != "sites":
            raise SourceValidationError(f"Invalid SharePoint site URL format: {uri}")

        # Extract site URL (up to /sites/sitename)
        site_name = path_parts[1]
        site_url = f"{parsed.scheme}://{parsed.netloc}/sites/{site_name}"

        # Extract library path (everything after site)
        if len(path_parts) > 2:
            library_path = "/".join(path_parts[2:])
        else:
            library_path = "Shared Documents"  # Default document library

        return site_url, library_path

    def _extract_tenant_url(self, site_url: str) -> str:
        """Extract tenant URL from site URL."""
        parsed = urlparse(site_url)
        return f"{parsed.scheme}://{parsed.netloc}"

    async def validate(self) -> None:
        """Validate SharePoint site access and authentication."""
        try:
            # Authenticate
            await self._authenticate()

            # Get site information
            await self._get_site_info()

            # Get library information
            await self._get_library_info()

            self._validated = True
            self.logger.info(f"Validated SharePoint source: {self.site_url}/{self.library_path}")

        except Exception as e:
            raise SourceValidationError(f"SharePoint validation failed: {e}")

    async def _authenticate(self) -> None:
        """Authenticate with SharePoint using configured credentials."""
        creds = self.config.credentials

        if not creds:
            raise SourceValidationError("SharePoint credentials are required")

        auth_method = creds.get("auth_method", "client_credentials")

        if auth_method == "client_credentials":
            await self._authenticate_client_credentials(creds)
        elif auth_method == "device_code":
            await self._authenticate_device_code(creds)
        elif auth_method == "username_password":
            await self._authenticate_username_password(creds)
        else:
            raise SourceValidationError(f"Unsupported authentication method: {auth_method}")

    async def _authenticate_client_credentials(self, creds: Dict[str, Any]) -> None:
        """Authenticate using client credentials (app-only)."""
        required_fields = ["client_id", "client_secret", "tenant_id"]
        missing = [field for field in required_fields if field not in creds]
        if missing:
            raise SourceValidationError(f"Missing required credentials for client_credentials: {missing}")

        app = ConfidentialClientApplication(
            client_id=creds["client_id"],
            client_credential=creds["client_secret"],
            authority=f"https://login.microsoftonline.com/{creds['tenant_id']}",
        )

        # Request token for SharePoint
        scopes = [f"{self.tenant_url}/.default"]
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: app.acquire_token_for_client(scopes=scopes)
        )

        if "access_token" not in result:
            error = result.get("error_description", result.get("error", "Unknown error"))
            raise SourceValidationError(f"Failed to acquire access token: {error}")

        self.access_token = result["access_token"]
        self.token_expires_at = datetime.now().timestamp() + result.get("expires_in", 3600)

        # Set authorization header
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    async def _authenticate_device_code(self, creds: Dict[str, Any]) -> None:
        """Authenticate using device code flow (interactive)."""
        required_fields = ["client_id", "tenant_id"]
        missing = [field for field in required_fields if field not in creds]
        if missing:
            raise SourceValidationError(f"Missing required credentials for device_code: {missing}")

        app = PublicClientApplication(
            client_id=creds["client_id"], authority=f"https://login.microsoftonline.com/{creds['tenant_id']}"
        )

        scopes = ["https://graph.microsoft.com/Sites.Read.All"]

        # Start device code flow
        flow = await asyncio.get_event_loop().run_in_executor(None, lambda: app.initiate_device_flow(scopes=scopes))

        if "user_code" not in flow:
            raise SourceValidationError("Failed to initiate device code flow")

        # Display device code to user
        print(f"\nTo authenticate with SharePoint, visit: {flow['verification_uri']}")
        print(f"Enter the code: {flow['user_code']}")
        print("Waiting for authentication...")

        # Wait for user to complete authentication
        result = await asyncio.get_event_loop().run_in_executor(None, lambda: app.acquire_token_by_device_flow(flow))

        if "access_token" not in result:
            error = result.get("error_description", result.get("error", "Unknown error"))
            raise SourceValidationError(f"Failed to acquire access token: {error}")

        self.access_token = result["access_token"]
        self.token_expires_at = datetime.now().timestamp() + result.get("expires_in", 3600)

        # Set authorization header
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    async def _authenticate_username_password(self, creds: Dict[str, Any]) -> None:
        """Authenticate using username/password (not recommended for production)."""
        required_fields = ["client_id", "tenant_id", "username", "password"]
        missing = [field for field in required_fields if field not in creds]
        if missing:
            raise SourceValidationError(f"Missing required credentials for username_password: {missing}")

        app = PublicClientApplication(
            client_id=creds["client_id"], authority=f"https://login.microsoftonline.com/{creds['tenant_id']}"
        )

        scopes = ["https://graph.microsoft.com/Sites.Read.All"]

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: app.acquire_token_by_username_password(
                username=creds["username"], password=creds["password"], scopes=scopes
            ),
        )

        if "access_token" not in result:
            error = result.get("error_description", result.get("error", "Unknown error"))
            raise SourceValidationError(f"Failed to acquire access token: {error}")

        self.access_token = result["access_token"]
        self.token_expires_at = datetime.now().timestamp() + result.get("expires_in", 3600)

        # Set authorization header
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    async def _get_site_info(self) -> None:
        """Get SharePoint site information."""
        # Extract site path from URL
        parsed = urlparse(self.site_url)
        site_path = parsed.path

        # Use Microsoft Graph API to get site info
        graph_url = f"https://graph.microsoft.com/v1.0/sites/{parsed.netloc}:{site_path}"

        response = await asyncio.get_event_loop().run_in_executor(None, lambda: self.session.get(graph_url))

        if response.status_code != 200:
            raise SourceValidationError(f"Failed to get site info: {response.status_code} - {response.text}")

        site_data = response.json()
        self.site_id = site_data["id"]

        self.logger.debug(f"Found SharePoint site: {site_data.get('displayName', 'Unknown')}")

    async def _get_library_info(self) -> None:
        """Get document library information."""
        # Get all document libraries for the site
        graph_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/lists"
        params = {"$filter": "list/template eq 'documentLibrary'"}

        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.session.get(graph_url, params=params)
        )

        if response.status_code != 200:
            raise SourceValidationError(f"Failed to get document libraries: {response.status_code} - {response.text}")

        libraries = response.json().get("value", [])

        # Find the target library
        library_name = self.library_path.split("/")[0]

        for library in libraries:
            if library["name"] == library_name or library["displayName"] == library_name:
                self.library_id = library["id"]
                break

        if not self.library_id:
            available = [lib["displayName"] for lib in libraries]
            raise SourceValidationError(
                f"Document library '{library_name}' not found. Available libraries: {available}"
            )

        self.logger.debug(f"Found document library: {library_name}")

    async def list_documents(self) -> List[SourceDocument]:
        """List all documents in the SharePoint library."""
        if not self._validated:
            await self.validate()

        documents = []

        try:
            # Get folder path within library
            folder_path = "/".join(self.library_path.split("/")[1:]) if "/" in self.library_path else ""

            # Recursively get all items
            items = await self._get_library_items(folder_path)

            for item in items:
                try:
                    if item.get("file"):  # It's a file, not a folder
                        doc = self._create_document_from_sharepoint_item(item)
                        documents.append(doc)
                except Exception as e:
                    self.logger.warning(f"Error processing SharePoint item {item.get('name', 'unknown')}: {e}")
                    continue

        except Exception as e:
            raise SourceValidationError(f"Failed to list SharePoint documents: {e}")

        # Apply filtering
        filtered_documents = self._filter_documents(documents)

        self.logger.info(f"Found {len(documents)} total files, {len(filtered_documents)} after filtering")
        return filtered_documents

    async def _get_library_items(self, folder_path: str = "") -> List[Dict[str, Any]]:
        """Recursively get all items from SharePoint library."""
        items = []

        # Build URL for the specific folder
        if folder_path:
            graph_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/lists/{self.library_id}/items"
            params = {"$expand": "fields,driveItem", "$filter": f"fields/FileDirRef eq '{folder_path}'"}
        else:
            graph_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/lists/{self.library_id}/items"
            params = {"$expand": "fields,driveItem"}

        # Handle pagination
        next_url = graph_url
        while next_url:
            if next_url == graph_url:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.session.get(next_url, params=params)
                )
            else:
                response = await asyncio.get_event_loop().run_in_executor(None, lambda: self.session.get(next_url))

            if response.status_code != 200:
                raise SourceValidationError(f"Failed to get library items: {response.status_code} - {response.text}")

            data = response.json()
            page_items = data.get("value", [])
            items.extend(page_items)

            # Check for next page
            next_url = data.get("@odata.nextLink")

            # Safety check
            if len(items) > 10000:
                self.logger.warning("Limiting to first 10000 items from SharePoint")
                break

        return items

    async def get_document(self, document: SourceDocument) -> SourceDocument:
        """Retrieve document content from SharePoint."""
        try:
            download_url = document.metadata.get("download_url")
            if not download_url:
                raise SourceValidationError(f"No download URL available for document: {document.name}")

            # Download file content
            response = await asyncio.get_event_loop().run_in_executor(None, lambda: self.session.get(download_url))

            if response.status_code != 200:
                raise SourceValidationError(f"Failed to download document: {response.status_code} - {response.text}")

            # Update document with content
            document.content = response.content
            document.size = len(response.content)

            return document

        except Exception as e:
            raise SourceValidationError(f"Failed to retrieve SharePoint document {document.name}: {e}")

    def _create_document_from_sharepoint_item(self, item: Dict[str, Any]) -> SourceDocument:
        """Create a SourceDocument from a SharePoint list item."""
        fields = item.get("fields", {})
        drive_item = item.get("driveItem", {})

        name = fields.get("FileLeafRef") or drive_item.get("name", "unknown")

        # Get file size
        size = drive_item.get("size", 0)

        # Get last modified date
        last_modified = None
        if "lastModifiedDateTime" in drive_item:
            last_modified = datetime.fromisoformat(drive_item["lastModifiedDateTime"].replace("Z", "+00:00"))
        elif "Modified" in fields:
            last_modified = datetime.fromisoformat(fields["Modified"].replace("Z", "+00:00"))

        # Build source path
        file_ref = fields.get("FileRef", f"/{name}")
        source_path = f"{self.site_url}{file_ref}"

        # Get download URL
        download_url = drive_item.get("@microsoft.graph.downloadUrl")

        return SourceDocument(
            name=name,
            source_path=source_path,
            content_type="",  # Will be inferred
            size=size,
            last_modified=last_modified,
            metadata={
                "sharepoint_item_id": item.get("id"),
                "file_ref": file_ref,
                "download_url": download_url,
                "author": fields.get("Author", {}).get("LookupValue", "Unknown"),
                "created": fields.get("Created"),
                "modified": fields.get("Modified"),
                "version": fields.get("_UIVersionString", "1.0"),
            },
        )
