/**
        * Copyright (c) 2017 LIBBLE team supervised by Dr. Wu-Jun LI at Nanjing University.
        * All Rights Reserved.
        * Licensed under the Apache License, Version 2.0 (the "License");
        * you may not use this file except in compliance with the License.
        * You may obtain a copy of the License at
        *
        * http://www.apache.org/licenses/LICENSE-2.0
        *
        * Unless required by applicable law or agreed to in writing, software
        * distributed under the License is distributed on an "AS IS" BASIS,
        * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        * See the License for the specific language governing permissions and
        * limitations under the License. */

#ifndef _PROTOCOL_HPP_
#define _PROTOCOL_HPP_

/* this file define the tag number used for MPI for sending different messages */

// from coordinator
#define CW_PARAMS 100     // coordinator sends parameters to worker
#define CW_INFO 101       // coordinator sends info to worker
#define CW_GRAD 102       // coordinator sends full grad to worker
#define CS_INFO 103       // coordinator sends info to server
#define CSWPULL_INFO 104  // coordinator sends pull w_id info to server
#define CSWPUSH_INFO 105  // coordinator sends push w_id info to server

// from server
#define SW_PARAMS 200  // server sends parameters to worker
#define SC_EPOCH 201   // server sends epoch to coordinator to count time
#define SC_PARAMS 202  // server sends parameters to coordinator in the end
#define SW_C 203       // server sends c to worker
#define SW_GRAD 204    // server sends full grad to worker

// from worker
#define WS_GRADS 300   // worker sends gradients to server
#define WC_LOSS 301    // worker sends loss to coordinator
#define WC_GRAD 302    // worker sends part full grad to coordinator
#define WC_PARAMS 303  // worker sends parameters to coordinator
#define WS_PARAMS 304  // worker sends parameters to coordinator
#define WCP_INFO 305   // worker sends pull info to coordinator
#define WCG_INFO 306   // worker sends push info to coordinator
#define WS_C 307       // worker sends c to server
#define WC_ACCU 308    // worker sends accuracy to coordinator

#endif