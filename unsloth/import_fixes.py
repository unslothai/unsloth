# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def fix_message_factory_issue():
    # Fix up AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
    # MUST do this at the start primarily due to tensorflow causing issues
    try:
        import google.protobuf.message_factory
        class MessageFactory:
            def CreatePrototype(self, *args, **kwargs): return
            def GetMessages(self, *args, **kwargs): return
            def GetPrototype(self, *args, **kwargs): return
        if not hasattr(google.protobuf.message_factory, "MessageFactory"):
            google.protobuf.message_factory.MessageFactory = MessageFactory
        elif hasattr(google.protobuf.message_factory, "MessageFactory") and \
            not hasattr(google.protobuf.message_factory.MessageFactory, "GetPrototype") and \
            not hasattr(google.protobuf.message_factory, "GetMessageClass"):
            google.protobuf.message_factory.MessageFactory = MessageFactory
        elif hasattr(google.protobuf.message_factory, "MessageFactory") and \
            not hasattr(google.protobuf.message_factory.MessageFactory, "GetPrototype") and \
            hasattr(google.protobuf.message_factory, "GetMessageClass"):
            GetMessageClass = google.protobuf.message_factory.GetMessageClass
            def GetPrototype(self, descriptor):
                return GetMessageClass(descriptor)
            google.protobuf.message_factory.MessageFactory.GetPrototype = GetPrototype
        pass
    except:
        pass
pass
