"""
Amazon S3 input source implementation.
"""

import asyncio
from typing import List, Optional
from datetime import datetime
from urllib.parse import urlparse
import re

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from .base import BaseInputSource, SourceDocument, SourceValidationError


class S3InputSource(BaseInputSource):
    """Input source for Amazon S3 buckets."""
    
    def __init__(self, config):
        """Initialize S3 input source."""
        super().__init__(config)
        
        if not BOTO3_AVAILABLE:
            raise SourceValidationError(
                "boto3 is required for S3 input sources. Install with: pip install boto3"
            )
        
        # Parse S3 URI
        self.bucket_name, self.prefix = self._parse_s3_uri(config.source_uri)
        
        # Initialize S3 client
        self.s3_client = None
        self._setup_s3_client()
    
    def _parse_s3_uri(self, uri: str) -> tuple[str, str]:
        """Parse S3 URI into bucket and prefix."""
        # Support formats: s3://bucket/prefix, s3://bucket, bucket/prefix, bucket
        if uri.startswith('s3://'):
            uri = uri[5:]  # Remove s3:// prefix
        
        parts = uri.split('/', 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        
        # Validate bucket name
        if not self._is_valid_bucket_name(bucket):
            raise SourceValidationError(f"Invalid S3 bucket name: {bucket}")
        
        return bucket, prefix
    
    def _is_valid_bucket_name(self, bucket_name: str) -> bool:
        """Validate S3 bucket name according to AWS rules."""
        if not bucket_name or len(bucket_name) < 3 or len(bucket_name) > 63:
            return False
        
        # Check pattern: lowercase letters, numbers, dots, hyphens
        if not re.match(r'^[a-z0-9.-]+$', bucket_name):
            return False
        
        # Cannot start/end with dot or hyphen
        if bucket_name.startswith('.') or bucket_name.startswith('-'):
            return False
        if bucket_name.endswith('.') or bucket_name.endswith('-'):
            return False
        
        # Cannot have consecutive dots
        if '..' in bucket_name:
            return False
        
        return True
    
    def _setup_s3_client(self):
        """Setup S3 client with credentials from config."""
        session_kwargs = {}
        
        # Use credentials from config if provided
        if self.config.credentials:
            if 'aws_access_key_id' in self.config.credentials:
                session_kwargs['aws_access_key_id'] = self.config.credentials['aws_access_key_id']
            if 'aws_secret_access_key' in self.config.credentials:
                session_kwargs['aws_secret_access_key'] = self.config.credentials['aws_secret_access_key']
            if 'aws_session_token' in self.config.credentials:
                session_kwargs['aws_session_token'] = self.config.credentials['aws_session_token']
            if 'region_name' in self.config.credentials:
                session_kwargs['region_name'] = self.config.credentials['region_name']
        
        try:
            session = boto3.Session(**session_kwargs)
            self.s3_client = session.client('s3')
        except Exception as e:
            raise SourceValidationError(f"Failed to create S3 client: {e}")
    
    async def validate(self) -> None:
        """Validate S3 bucket access and credentials."""
        try:
            # Test bucket access by listing objects with limit
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=self.prefix,
                    MaxKeys=1
                )
            )
            
            self._validated = True
            self.logger.info(f"Validated S3 source: s3://{self.bucket_name}/{self.prefix}")
            
        except NoCredentialsError:
            raise SourceValidationError(
                "AWS credentials not found. Configure credentials via AWS CLI, "
                "environment variables, or provide them in the source configuration."
            )
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                raise SourceValidationError(f"S3 bucket does not exist: {self.bucket_name}")
            elif error_code == 'AccessDenied':
                raise SourceValidationError(f"Access denied to S3 bucket: {self.bucket_name}")
            else:
                raise SourceValidationError(f"S3 validation failed: {e}")
        except Exception as e:
            raise SourceValidationError(f"Unexpected error during S3 validation: {e}")
    
    async def list_documents(self) -> List[SourceDocument]:
        """List all documents in the S3 bucket/prefix."""
        if not self._validated:
            await self.validate()
        
        documents = []
        continuation_token = None
        
        try:
            while True:
                # Prepare list_objects_v2 parameters
                list_params = {
                    'Bucket': self.bucket_name,
                    'Prefix': self.prefix,
                    'MaxKeys': self.config.batch_size
                }
                
                if continuation_token:
                    list_params['ContinuationToken'] = continuation_token
                
                # List objects
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.s3_client.list_objects_v2(**list_params)
                )
                
                # Process objects
                for obj in response.get('Contents', []):
                    # Skip directories (objects ending with /)
                    if obj['Key'].endswith('/'):
                        continue
                    
                    try:
                        doc = self._create_document_from_s3_object(obj)
                        documents.append(doc)
                    except Exception as e:
                        self.logger.warning(f"Error processing S3 object {obj['Key']}: {e}")
                        continue
                
                # Check for more objects
                if not response.get('IsTruncated', False):
                    break
                
                continuation_token = response.get('NextContinuationToken')
                
                # Safety check to prevent infinite loops
                if len(documents) > 10000:  # Configurable limit
                    self.logger.warning(f"Limiting to first 10000 objects from S3")
                    break
        
        except Exception as e:
            raise SourceValidationError(f"Failed to list S3 objects: {e}")
        
        # Apply filtering
        filtered_documents = self._filter_documents(documents)
        
        self.logger.info(f"Found {len(documents)} total objects, {len(filtered_documents)} after filtering")
        return filtered_documents
    
    async def get_document(self, document: SourceDocument) -> SourceDocument:
        """Retrieve document content from S3."""
        try:
            # Download object content
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=document.metadata['s3_key']
                )
            )
            
            # Read content
            content = response['Body'].read()
            
            # Update document with content
            document.content = content
            document.size = len(content)
            
            # Update metadata with response info
            document.metadata.update({
                'content_type': response.get('ContentType', ''),
                'etag': response.get('ETag', '').strip('"'),
                'last_modified': response.get('LastModified'),
                'cache_control': response.get('CacheControl', ''),
                'content_encoding': response.get('ContentEncoding', ''),
            })
            
            return document
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise SourceValidationError(f"S3 object not found: {document.metadata['s3_key']}")
            elif error_code == 'AccessDenied':
                raise SourceValidationError(f"Access denied to S3 object: {document.metadata['s3_key']}")
            else:
                raise SourceValidationError(f"Failed to download S3 object: {e}")
        except Exception as e:
            raise SourceValidationError(f"Unexpected error downloading S3 object: {e}")
    
    def _create_document_from_s3_object(self, s3_object: dict) -> SourceDocument:
        """Create a SourceDocument from an S3 object metadata."""
        key = s3_object['Key']
        name = key.split('/')[-1]  # Get filename from key
        
        # Skip empty filenames (shouldn't happen with proper filtering)
        if not name:
            raise ValueError(f"Invalid S3 object key: {key}")
        
        return SourceDocument(
            name=name,
            source_path=f"s3://{self.bucket_name}/{key}",
            content_type="",  # Will be inferred
            size=s3_object.get('Size', 0),
            last_modified=s3_object.get('LastModified'),
            metadata={
                's3_key': key,
                's3_bucket': self.bucket_name,
                'etag': s3_object.get('ETag', '').strip('"'),
                'storage_class': s3_object.get('StorageClass', 'STANDARD'),
            }
        )